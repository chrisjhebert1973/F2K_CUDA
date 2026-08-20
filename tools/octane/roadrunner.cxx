// Roadrunner — an SGI Octane (IRIX / MIPSpro / ViewKit) front-end for the
// F2K_CUDA image generator running on the DGX Spark ("sparky").
//
// A ViewKit/Motif app: type a prompt, pick model / resolution / steps / seed /
// batch count, hit Generate. The request goes over the LAN to octane_bridge on
// the Spark, which drives the resident CUDA worker and streams back progress
// plus raw RGB. We wrap that RGB in an XImage and blit it into an XmDrawingArea.
// No JSON, no image libraries, no OpenGL on this side — just Xlib.
//
//   build:  make            (see Makefile — MIPSpro CC + ViewKit + Motif)
//   run:    ./roadrunner -host sparky -port 1974
//           (or set F2K_BRIDGE_HOST / F2K_BRIDGE_PORT)
//
// ---- wire protocol (must match tools/octane_bridge.cpp) --------------------
// Request  (ASCII, two newline-terminated lines):
//     "GEN <res> <steps> <seed> <count> <model>\n"   ints; seed<0 => randomise;
//                                             model token, "-" => bridge default
//     "<prompt>\n"
//   ("LIST\n" instead returns newline-separated model names, then EOF.)
// Response: a stream of tagged messages (every int32 network byte order — this
// box is big-endian MIPS, the Spark is little-endian). Each begins with the
// 4-byte magic 'F','2','K','1' then an int32 type:
//     PROGRESS (1): u32 imgIndex, imgTotal, permille(0..1000), len, phase[len]
//     IMAGE    (2): u32 imgIndex, w, h, seed, then w*h*3 RGB bytes (top row first)
//     DONE     (3): u32 count
//     ERROR    (4): u32 len, message[len]

#include <Vk/VkApp.h>
#include <Vk/VkSimpleWindow.h>

#include <Xm/Xm.h>
#include <Xm/MainW.h>
#include <Xm/Form.h>
#include <Xm/Frame.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/CascadeB.h>
#include <Xm/RowColumn.h>
#include <Xm/Scale.h>
#include <Xm/Separator.h>
#include <Xm/Text.h>
#include <Xm/TextF.h>
#include <Xm/ToggleB.h>
#include <Xm/DrawingA.h>
#include <Xm/FileSB.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Real PNG output with no system libs (stb rolls its own deflate). Bundled in
// this directory so the Octane build is self-contained. If MIPSpro's C++ front
// end fights the header, compile it instead as C into a stbiw.o (see README).
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"          // decode a loaded init image (remix mode)

// -- config -------------------------------------------------------------------
static const char* DEFAULT_HOST = "sparky";
static const int   DEFAULT_PORT = 1974;
static const int   RES_CHOICES[]   = { 256, 512, 768, 1024 };
static const int   N_RES = 4;
static const int   COUNT_CHOICES[] = { 1, 2, 4, 8 };
static const int   N_COUNT = 4;
static const int   MAX_BATCH = 8;      // largest count we accept
static const int   THUMB = 104;        // thumbnail edge, px
static const int   INIT_MAX = 1024;    // cap the remix init image we ship
static const float MIN_ZOOM = 0.05f;
static const float MAX_ZOOM = 8.0f;
static const int   MAX_DISP = 3072;    // cap the scaled display image edge (memory)

// tagged message types (must match octane_bridge.cpp)
enum { MSG_PROGRESS = 1, MSG_IMAGE = 2, MSG_DONE = 3, MSG_ERROR = 4 };

// one generated image, held for the batch strip + save
struct BatchImg {
    unsigned char* rgb;        // owned copy of the raw RGB
    int            w, h;
    unsigned long  seed;
    XImage*        thumb;      // small XImage for the strip (owned)
};

// ============================================================================
class RoadrunnerWindow : public VkSimpleWindow {
public:
    RoadrunnerWindow(const char* name, const char* host, int port);
    virtual ~RoadrunnerWindow();
    virtual const char* className() { return "RoadrunnerWindow"; }

private:
    // controls
    Widget _prompt;                 // scrolled XmText (multi-line)
    Widget _seed;                   // XmTextField
    Widget _steps;                  // XmScale
    Widget _resToggles[N_RES];      // XmToggleButtons in a radio box
    Widget _generate;               // XmPushButton
    Widget _status;                 // XmLabel
    Widget _canvas;                 // XmDrawingArea (main image)
    Widget _progDA;                 // XmDrawingArea (progress bar)
    Widget _prevBtn, _nextBtn;      // batch navigation
    Widget _thumbDA[MAX_BATCH];     // thumbnail cells (pre-created, shown as used)
    Widget _strength;               // XmScale (remix strength, 5..100)
    Widget _loadBtn;                // "Load init image..." (remix)
    Widget _negative;               // XmTextField (negative prompt, needs CFG>1)
    Widget _cfg;                    // XmScale (guidance 1..10; 1 = off)
    Widget _var;                    // XmScale (variation %, 0..100)

    // network target / current selections
    char   _host[256];
    int    _port;
    int    _res;
    int    _count;                  // batch size
    int    _mode;                   // 0 = Generate, 1 = Remix
    char   _model[128];             // "(default)" => "-"

    // remix init image (already centre-cropped to square, RGB)
    unsigned char* _initRGB;
    int            _initDim;        // square edge, px
    int            _hasInit;

    // display state
    Display* _dpy;
    GC       _gc;
    int      _progPermille;         // 0..1000 for the progress bar

    // shown image + zoom/pan viewport
    unsigned char* _curRGB;         // owned copy of the shown image's RGB
    int      _curW, _curH;          // its native size
    float    _zoom;                 // display scale
    int      _offX, _offY;          // image top-left in canvas coords (pan)
    int      _fitMode;              // 1 = re-fit on resize / new image
    XImage*  _disp;                 // cached scaled image actually blitted (owned)
    int      _dispW, _dispH;
    int      _dragging, _dragX0, _dragY0, _offX0, _offY0;   // pan-drag state
    Pixmap   _buf;                  // off-screen double buffer for the canvas
    int      _bufW, _bufH;

    // current batch
    BatchImg _batch[MAX_BATCH];
    int      _batchN;               // images received so far this batch
    int      _selected;             // index shown in the main canvas, or -1

    // request snapshot (batch-wide) for File > Save; per-image seed is in _batch
    char*  _lastPrompt;
    char   _lastModelSel[128];
    int    _lastRes, _lastSteps;

    // async receive (XtAppAddInput-driven, tagged-message parser)
    int        _fd;
    XtInputId  _inputId;
    unsigned char* _rx;             // accumulation buffer (reused across runs)
    int        _rxLen;              // bytes appended
    int        _rxCap;
    int        _rxHead;             // parse cursor into _rx

    // -- callbacks (static trampolines -> instance methods) --
    static void generateCB(Widget, XtPointer, XtPointer);
    static void resCB(Widget, XtPointer, XtPointer);
    static void modelCB(Widget, XtPointer, XtPointer);
    static void countCB(Widget, XtPointer, XtPointer);
    static void modeCB(Widget, XtPointer, XtPointer);
    static void loadCB(Widget, XtPointer, XtPointer);
    static void loadOkCB(Widget, XtPointer, XtPointer);
    static void pasteCB(Widget, XtPointer, XtPointer);
    static void outpaintCB(Widget, XtPointer, XtPointer);
    static void exposeCB(Widget, XtPointer, XtPointer);
    static void resizeCB(Widget, XtPointer, XtPointer);
    static void canvasEH(Widget, XtPointer, XEvent*, Boolean*);
    static void zoomCB(Widget, XtPointer, XtPointer);
    static void progExposeCB(Widget, XtPointer, XtPointer);
    static void thumbExposeCB(Widget, XtPointer, XtPointer);
    static void thumbInputCB(Widget, XtPointer, XtPointer);
    static void prevCB(Widget, XtPointer, XtPointer);
    static void nextCB(Widget, XtPointer, XtPointer);
    static void inputCB(XtPointer, int*, XtInputId*);
    static void saveCB(Widget, XtPointer, XtPointer);
    static void saveOkCB(Widget, XtPointer, XtPointer);
    static void saveCancelCB(Widget, XtPointer, XtPointer);
    static void quitCB(Widget, XtPointer, XtPointer);

    void onGenerate();
    void onOutpaint();
    void armReceive(int fd);        // go non-blocking + hand socket to the event loop
    void onInput();
    void onSave();
    void doSave(const char* base);
    void onLoad();
    void doLoad(const char* path);
    void setMode(int mode);

    // helpers
    void    setStatus(const char* msg);
    int     connectBridge();
    int     queryModels(char names[][64], int maxN);
    int     appendRx(const unsigned char* d, int n);
    void    finishBatch(const char* status);
    XImage* makeScaledXImage(const unsigned char* rgb, int sw, int sh, int dw, int dh);
    void    showImage(int w, int h, const unsigned char* rgb);
    void    redraw();
    void    canvasSize(int* cw, int* ch);
    void    buildDisp();            // (re)build the scaled image for the current zoom
    void    fitToWindow();          // zoom so the whole image is visible, centred
    void    zoomTo(float nz, int cx, int cy);   // zoom, keeping (cx,cy) fixed
    void    onCanvasResize();
    void    onCanvasEvent(XEvent* ev);          // pan-drag
    void    clearBatch();
    void    addBatchImage(int idx, int w, int h, unsigned long seed, const unsigned char* rgb);
    void    selectImage(int k);
    void    redrawThumb(int i);
    void    onProgress(int idx, int total, int permille, const char* phase);
    void    drawProgress();
};

// -- blocking write of exactly n bytes (request side is tiny) -----------------
static int send_all(int fd, const void* buf, int n) {
    const char* p = (const char*)buf;
    while (n > 0) {
        int k = write(fd, p, n);
        if (k <= 0) return 0;
        p += k; n -= k;
    }
    return 1;
}

// ============================================================================
RoadrunnerWindow::RoadrunnerWindow(const char* name, const char* host, int port)
    : VkSimpleWindow(name)
{
    strncpy(_host, host, sizeof(_host) - 1);
    _host[sizeof(_host) - 1] = '\0';
    _port  = port;
    _res   = 512;
    _count = 1;
    _mode  = 0;
    _gc    = NULL;
    _dpy   = NULL;
    _progPermille = 0;
    _model[0] = '\0';
    _initRGB = NULL; _initDim = 0; _hasInit = 0;
    _curRGB = NULL; _curW = 0; _curH = 0;
    _zoom = 1.0f; _offX = 0; _offY = 0; _fitMode = 1;
    _disp = NULL; _dispW = 0; _dispH = 0;
    _dragging = 0; _dragX0 = _dragY0 = _offX0 = _offY0 = 0;
    _buf = 0; _bufW = 0; _bufH = 0;
    _fd = -1; _inputId = 0;
    _rx = NULL; _rxLen = 0; _rxCap = 0; _rxHead = 0;
    _batchN = 0; _selected = -1;
    for (int i = 0; i < MAX_BATCH; i++) {
        _batch[i].rgb = NULL; _batch[i].thumb = NULL;
        _batch[i].w = _batch[i].h = 0; _batch[i].seed = 0;
    }
    _lastPrompt = NULL; _lastModelSel[0] = '\0'; _lastRes = 0; _lastSteps = 0;

    Widget parent = mainWindowWidget();

    // ---- menu bar: File > Save / Quit --------------------------------------
    Widget menubar = XmCreateMenuBar(parent, (char*)"menubar", NULL, 0);
    XtManageChild(menubar);
    Widget filePD = XmCreatePulldownMenu(menubar, (char*)"filePD", NULL, 0);
    XmString fstr = XmStringCreateLocalized((char*)"File");
    XtVaCreateManagedWidget("File", xmCascadeButtonWidgetClass, menubar,
        XmNlabelString, fstr, XmNsubMenuId, filePD, NULL);
    XmStringFree(fstr);
    Widget saveItem = XtVaCreateManagedWidget("Save Image + Params...",
        xmPushButtonWidgetClass, filePD, NULL);
    XtAddCallback(saveItem, XmNactivateCallback, &RoadrunnerWindow::saveCB, (XtPointer)this);
    Widget opItem = XtVaCreateManagedWidget("Outpaint shown image",
        xmPushButtonWidgetClass, filePD, NULL);
    XtAddCallback(opItem, XmNactivateCallback, &RoadrunnerWindow::outpaintCB, (XtPointer)this);
    XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, filePD, NULL);
    Widget quitItem = XtVaCreateManagedWidget("Quit", xmPushButtonWidgetClass, filePD, NULL);
    XtAddCallback(quitItem, XmNactivateCallback, &RoadrunnerWindow::quitCB, (XtPointer)this);

    // Root form: a control column on the left, the image canvas filling the rest.
    Widget form = XmCreateForm(parent, (char*)"form", NULL, 0);
    Widget panel = XtVaCreateManagedWidget("panel", xmFormWidgetClass, form,
        XmNtopAttachment,    XmATTACH_FORM,
        XmNbottomAttachment, XmATTACH_FORM,
        XmNleftAttachment,   XmATTACH_FORM,
        XmNwidth,            440,           // fits 4 thumbnails per row at THUMB px
        NULL);

    Widget promptLbl = XtVaCreateManagedWidget("Prompt:", xmLabelWidgetClass, panel,
        XmNtopAttachment,  XmATTACH_FORM,
        XmNleftAttachment, XmATTACH_FORM,
        XmNalignment,      XmALIGNMENT_BEGINNING,
        NULL);

    // multi-line, editable, word-wrapped prompt box. NB: XmNwordWrap only takes
    // effect when horizontal scrolling is OFF, hence XmNscrollHorizontal False.
    Arg args[12]; int n = 0;
    XtSetArg(args[n], XmNeditMode, XmMULTI_LINE_EDIT); n++;
    XtSetArg(args[n], XmNwordWrap, True);              n++;
    XtSetArg(args[n], XmNscrollHorizontal, False);     n++;
    XtSetArg(args[n], XmNrows, 6);                     n++;
    XtSetArg(args[n], XmNcolumns, 32);                 n++;
    _prompt = XmCreateScrolledText(panel, (char*)"prompt", args, n);
    XmTextSetString(_prompt, (char*)"a black and white Akita husky dog "
                                    "sitting on a race car, cinematic");
    XtManageChild(_prompt);
    Widget promptSW = XtParent(_prompt);      // ScrolledText's geometry parent
    XtVaSetValues(promptSW,
        XmNtopAttachment,    XmATTACH_WIDGET, XmNtopWidget, promptLbl,
        XmNleftAttachment,   XmATTACH_FORM,
        XmNrightAttachment,  XmATTACH_FORM,
        NULL);

    // Paste button (X CLIPBOARD, i.e. Ctrl-C'd text). The middle mouse button
    // still pastes the PRIMARY selection anywhere in the field, the classic X way.
    Widget pasteBtn = XtVaCreateManagedWidget("Paste", xmPushButtonWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, promptSW,
        XmNleftAttachment,  XmATTACH_FORM,
        NULL);
    XtAddCallback(pasteBtn, XmNactivateCallback, &RoadrunnerWindow::pasteCB, (XtPointer)this);

    // ---- model option menu (populated from the bridge's LIST query) --------
    char models[32][64];
    int nModels = queryModels(models, 32);
    if (nModels <= 0) { strcpy(models[0], "(default)"); nModels = 1; }
    Widget modelPD = XmCreatePulldownMenu(panel, (char*)"modelPD", NULL, 0);
    Widget firstModelBtn = NULL;
    for (int i = 0; i < nModels; i++) {
        Widget b = XtVaCreateManagedWidget(models[i], xmPushButtonWidgetClass, modelPD, NULL);
        XtAddCallback(b, XmNactivateCallback, &RoadrunnerWindow::modelCB, (XtPointer)this);
        if (i == 0) firstModelBtn = b;
    }
    strncpy(_model, models[0], sizeof(_model) - 1);
    _model[sizeof(_model) - 1] = '\0';
    XmString mlbl = XmStringCreateLocalized((char*)"Model:");
    Arg ma[2]; int mn = 0;
    XtSetArg(ma[mn], XmNsubMenuId, modelPD);  mn++;
    XtSetArg(ma[mn], XmNlabelString, mlbl);   mn++;
    Widget modelOM = XmCreateOptionMenu(panel, (char*)"modelOM", ma, mn);
    XmStringFree(mlbl);
    XtVaSetValues(modelOM,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, pasteBtn,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        NULL);
    if (firstModelBtn) XtVaSetValues(modelOM, XmNmenuHistory, firstModelBtn, NULL);
    XtManageChild(modelOM);

    // ---- resolution radio box ----------------------------------------------
    Widget resLbl = XtVaCreateManagedWidget("Resolution:", xmLabelWidgetClass, panel,
        XmNtopAttachment,  XmATTACH_WIDGET, XmNtopWidget, modelOM,
        XmNleftAttachment, XmATTACH_FORM,
        XmNalignment,      XmALIGNMENT_BEGINNING,
        NULL);
    Widget resBox = XtVaCreateManagedWidget("resBox", xmRowColumnWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, resLbl,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        XmNorientation,     XmHORIZONTAL,
        XmNradioBehavior,   True,
        XmNpacking,         XmPACK_COLUMN,
        XmNnumColumns,      1,
        NULL);
    for (int i = 0; i < N_RES; i++) {
        char lbl[16]; sprintf(lbl, "%d", RES_CHOICES[i]);
        _resToggles[i] = XtVaCreateManagedWidget(lbl, xmToggleButtonWidgetClass, resBox,
            XmNset, (RES_CHOICES[i] == _res) ? True : False, NULL);
        XtAddCallback(_resToggles[i], XmNvalueChangedCallback,
                      &RoadrunnerWindow::resCB, (XtPointer)this);
    }

    // ---- steps scale -------------------------------------------------------
    Widget stepsLbl = XtVaCreateManagedWidget("Steps:", xmLabelWidgetClass, panel,
        XmNtopAttachment,  XmATTACH_WIDGET, XmNtopWidget, resBox,
        XmNleftAttachment, XmATTACH_FORM,
        XmNalignment,      XmALIGNMENT_BEGINNING,
        NULL);
    _steps = XtVaCreateManagedWidget("steps", xmScaleWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, stepsLbl,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        XmNorientation,     XmHORIZONTAL,
        XmNminimum, 1, XmNmaximum, 30, XmNvalue, 4, XmNshowValue, True,
        NULL);

    // ---- seed field --------------------------------------------------------
    Widget seedLbl = XtVaCreateManagedWidget("Seed (blank = random):",
        xmLabelWidgetClass, panel,
        XmNtopAttachment,  XmATTACH_WIDGET, XmNtopWidget, _steps,
        XmNleftAttachment, XmATTACH_FORM,
        XmNalignment,      XmALIGNMENT_BEGINNING,
        NULL);
    _seed = XtVaCreateManagedWidget("seed", xmTextFieldWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, seedLbl,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        NULL);

    // ---- batch count option menu -------------------------------------------
    Widget countPD = XmCreatePulldownMenu(panel, (char*)"countPD", NULL, 0);
    Widget firstCountBtn = NULL;
    for (int i = 0; i < N_COUNT; i++) {
        char lbl[8]; sprintf(lbl, "%d", COUNT_CHOICES[i]);
        Widget b = XtVaCreateManagedWidget(lbl, xmPushButtonWidgetClass, countPD, NULL);
        XtAddCallback(b, XmNactivateCallback, &RoadrunnerWindow::countCB, (XtPointer)this);
        if (i == 0) firstCountBtn = b;
    }
    XmString clbl = XmStringCreateLocalized((char*)"Batch:");
    Arg ca[2]; int cn = 0;
    XtSetArg(ca[cn], XmNsubMenuId, countPD); cn++;
    XtSetArg(ca[cn], XmNlabelString, clbl);  cn++;
    Widget countOM = XmCreateOptionMenu(panel, (char*)"countOM", ca, cn);
    XmStringFree(clbl);
    XtVaSetValues(countOM,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, _seed,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        NULL);
    if (firstCountBtn) XtVaSetValues(countOM, XmNmenuHistory, firstCountBtn, NULL);
    XtManageChild(countOM);

    // ---- mode option menu (Generate / Remix) -------------------------------
    Widget modePD = XmCreatePulldownMenu(panel, (char*)"modePD", NULL, 0);
    static const char* MODE_NAMES[] = { "Generate", "Remix" };
    Widget firstModeBtn = NULL;
    for (int i = 0; i < 2; i++) {
        Widget b = XtVaCreateManagedWidget(MODE_NAMES[i], xmPushButtonWidgetClass, modePD, NULL);
        XtAddCallback(b, XmNactivateCallback, &RoadrunnerWindow::modeCB, (XtPointer)this);
        if (i == 0) firstModeBtn = b;
    }
    XmString dlbl = XmStringCreateLocalized((char*)"Mode:");
    Arg da[2]; int dn = 0;
    XtSetArg(da[dn], XmNsubMenuId, modePD); dn++;
    XtSetArg(da[dn], XmNlabelString, dlbl); dn++;
    Widget modeOM = XmCreateOptionMenu(panel, (char*)"modeOM", da, dn);
    XmStringFree(dlbl);
    XtVaSetValues(modeOM,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, countOM,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        NULL);
    if (firstModeBtn) XtVaSetValues(modeOM, XmNmenuHistory, firstModeBtn, NULL);
    XtManageChild(modeOM);

    // ---- remix controls: Load init image + Strength (enabled in Remix mode) -
    _loadBtn = XtVaCreateManagedWidget("Load init image...", xmPushButtonWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, modeOM,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        NULL);
    XtAddCallback(_loadBtn, XmNactivateCallback, &RoadrunnerWindow::loadCB, (XtPointer)this);
    _strength = XtVaCreateManagedWidget("strength", xmScaleWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, _loadBtn,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        XmNorientation,     XmHORIZONTAL,
        XmNminimum, 5, XmNmaximum, 100, XmNvalue, 60, XmNshowValue, True,
        XmNtitleString, XmStringCreateLocalized((char*)"Strength %"),
        NULL);

    // ---- advanced: negative prompt, guidance (CFG), variation --------------
    Widget negLbl = XtVaCreateManagedWidget("Negative (needs Guidance > 1):",
        xmLabelWidgetClass, panel,
        XmNtopAttachment,  XmATTACH_WIDGET, XmNtopWidget, _strength,
        XmNleftAttachment, XmATTACH_FORM, XmNalignment, XmALIGNMENT_BEGINNING,
        NULL);
    _negative = XtVaCreateManagedWidget("negative", xmTextFieldWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, negLbl,
        XmNleftAttachment,  XmATTACH_FORM, XmNrightAttachment, XmATTACH_FORM,
        NULL);
    Widget advRow = XtVaCreateManagedWidget("adv", xmRowColumnWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, _negative,
        XmNleftAttachment,  XmATTACH_FORM, XmNrightAttachment, XmATTACH_FORM,
        XmNorientation,     XmHORIZONTAL, XmNpacking, XmPACK_COLUMN, XmNnumColumns, 1,
        NULL);
    // 10..30 with 1 decimal point shown => 1.0..3.0 (10 = off). klein is guidance-
    // distilled, so it over-cooks past ~2; 1..3 in 0.1 steps is the useful range.
    _cfg = XtVaCreateManagedWidget("cfg", xmScaleWidgetClass, advRow,
        XmNorientation, XmHORIZONTAL,
        XmNminimum, 10, XmNmaximum, 30, XmNvalue, 10, XmNdecimalPoints, 1, XmNshowValue, True,
        XmNtitleString, XmStringCreateLocalized((char*)"Guidance"),
        NULL);
    _var = XtVaCreateManagedWidget("var", xmScaleWidgetClass, advRow,
        XmNorientation, XmHORIZONTAL,
        XmNminimum, 0, XmNmaximum, 100, XmNvalue, 0, XmNshowValue, True,
        XmNtitleString, XmStringCreateLocalized((char*)"Variation %"),
        NULL);

    // ---- generate button ---------------------------------------------------
    _generate = XtVaCreateManagedWidget("Generate", xmPushButtonWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, advRow,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        NULL);
    XtAddCallback(_generate, XmNactivateCallback,
                  &RoadrunnerWindow::generateCB, (XtPointer)this);

    // ---- progress bar (a plain DrawingArea we fill) ------------------------
    _progDA = XtVaCreateManagedWidget("progress", xmDrawingAreaWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, _generate,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        XmNheight,          16,
        XmNbackground,      BlackPixelOfScreen(XtScreen(form)),
        NULL);
    XtAddCallback(_progDA, XmNexposeCallback,
                  &RoadrunnerWindow::progExposeCB, (XtPointer)this);

    // ---- batch navigation row ----------------------------------------------
    Widget navRow = XtVaCreateManagedWidget("nav", xmRowColumnWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, _progDA,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        XmNorientation,     XmHORIZONTAL,
        XmNpacking,         XmPACK_COLUMN,
        XmNnumColumns,      1,
        NULL);
    _prevBtn = XtVaCreateManagedWidget("<< Prev", xmPushButtonWidgetClass, navRow, NULL);
    _nextBtn = XtVaCreateManagedWidget("Next >>", xmPushButtonWidgetClass, navRow, NULL);
    XtAddCallback(_prevBtn, XmNactivateCallback, &RoadrunnerWindow::prevCB, (XtPointer)this);
    XtAddCallback(_nextBtn, XmNactivateCallback, &RoadrunnerWindow::nextCB, (XtPointer)this);
    XtSetSensitive(_prevBtn, False);
    XtSetSensitive(_nextBtn, False);

    // ---- zoom controls (SGI mice have no scroll wheel; drag canvas to pan) --
    Widget zoomRow = XtVaCreateManagedWidget("zoom", xmRowColumnWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, navRow,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        XmNorientation,     XmHORIZONTAL,
        XmNpacking,         XmPACK_COLUMN,
        XmNnumColumns,      1,
        NULL);
    static const char* ZOOM_LBL[] = { "Zoom-", "Fit", "1:1", "Zoom+" };
    for (int i = 0; i < 4; i++) {
        Widget b = XtVaCreateManagedWidget(ZOOM_LBL[i], xmPushButtonWidgetClass, zoomRow, NULL);
        XtAddCallback(b, XmNactivateCallback, &RoadrunnerWindow::zoomCB, (XtPointer)this);
    }

    // ---- thumbnail strip (2 rows of 4; cells shown as images arrive) -------
    Widget thumbStrip = XtVaCreateManagedWidget("thumbs", xmRowColumnWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, zoomRow,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        XmNorientation,     XmHORIZONTAL,
        XmNpacking,         XmPACK_COLUMN,
        XmNnumColumns,      2,               // 2 rows (Motif: rows when HORIZONTAL)
        NULL);
    for (int i = 0; i < MAX_BATCH; i++) {
        _thumbDA[i] = XtVaCreateWidget("thumb", xmDrawingAreaWidgetClass, thumbStrip,
            XmNwidth,  THUMB, XmNheight, THUMB,
            // A DrawingArea defaults to XmRESIZE_ANY and, having no child widgets,
            // collapses to a minimal size (~16px) — which the RowColumn then uses
            // for its cells. Pin the size so our THUMB dimensions actually stick.
            XmNresizePolicy, XmRESIZE_NONE,
            XmNbackground, BlackPixelOfScreen(XtScreen(form)),
            NULL);                            // created UNmanaged; shown when used
        XtAddCallback(_thumbDA[i], XmNexposeCallback,
                      &RoadrunnerWindow::thumbExposeCB, (XtPointer)this);
        XtAddCallback(_thumbDA[i], XmNinputCallback,
                      &RoadrunnerWindow::thumbInputCB, (XtPointer)this);
    }

    // ---- status line -------------------------------------------------------
    _status = XtVaCreateManagedWidget("statusLbl", xmLabelWidgetClass, panel,
        XmNbottomAttachment, XmATTACH_FORM,
        XmNleftAttachment,   XmATTACH_FORM,
        XmNrightAttachment,  XmATTACH_FORM,
        XmNalignment,        XmALIGNMENT_BEGINNING,
        NULL);
    setStatus("Ready.");

    // ---- image canvas (right) ----------------------------------------------
    Widget frame = XtVaCreateManagedWidget("frame", xmFrameWidgetClass, form,
        XmNtopAttachment,    XmATTACH_FORM,
        XmNbottomAttachment, XmATTACH_FORM,
        XmNrightAttachment,  XmATTACH_FORM,
        XmNleftAttachment,   XmATTACH_WIDGET, XmNleftWidget, panel,
        NULL);
    _canvas = XtVaCreateManagedWidget("canvas", xmDrawingAreaWidgetClass, frame,
        XmNwidth, 640, XmNheight, 640,
        XmNbackground, BlackPixelOfScreen(XtScreen(form)),
        NULL);
    XtAddCallback(_canvas, XmNexposeCallback, &RoadrunnerWindow::exposeCB, (XtPointer)this);
    XtAddCallback(_canvas, XmNresizeCallback, &RoadrunnerWindow::resizeCB, (XtPointer)this);
    XtAddEventHandler(_canvas, ButtonPressMask | ButtonReleaseMask | Button1MotionMask,
                      False, &RoadrunnerWindow::canvasEH, (XtPointer)this);

    XtManageChild(form);
    addView(form);
    XmMainWindowSetAreas(parent, menubar, NULL, NULL, NULL, form);
    _dpy = XtDisplay(form);
    setMode(0);                    // start in Generate mode (remix controls off)
}

RoadrunnerWindow::~RoadrunnerWindow() {
    if (_inputId) XtRemoveInput(_inputId);
    if (_fd >= 0) close(_fd);
    if (_rx)         free(_rx);
    if (_lastPrompt) free(_lastPrompt);
    if (_initRGB)    free(_initRGB);
    if (_curRGB)     free(_curRGB);
    clearBatch();
    if (_disp) XDestroyImage(_disp);
    if (_buf)  XFreePixmap(_dpy, _buf);
    if (_gc)   XFreeGC(_dpy, _gc);
}

// -- status label -------------------------------------------------------------
void RoadrunnerWindow::setStatus(const char* msg) {
    XmString s = XmStringCreateLocalized((char*)msg);
    XtVaSetValues(_status, XmNlabelString, s, NULL);
    XmStringFree(s);
    XmUpdateDisplay(_status);          // paint immediately (we're event-driven)
}

// -- simple control callbacks -------------------------------------------------
void RoadrunnerWindow::resCB(Widget w, XtPointer client, XtPointer) {
    RoadrunnerWindow* self = (RoadrunnerWindow*)client;
    if (!XmToggleButtonGetState(w)) return;
    for (int i = 0; i < N_RES; i++)
        if (self->_resToggles[i] == w) { self->_res = RES_CHOICES[i]; return; }
}
void RoadrunnerWindow::modelCB(Widget w, XtPointer client, XtPointer) {
    RoadrunnerWindow* self = (RoadrunnerWindow*)client;
    strncpy(self->_model, XtName(w), sizeof(self->_model) - 1);
    self->_model[sizeof(self->_model) - 1] = '\0';
}
void RoadrunnerWindow::countCB(Widget w, XtPointer client, XtPointer) {
    ((RoadrunnerWindow*)client)->_count = atoi(XtName(w));
}
void RoadrunnerWindow::pasteCB(Widget, XtPointer client, XtPointer) {
    XmTextPaste(((RoadrunnerWindow*)client)->_prompt);   // paste X CLIPBOARD at cursor
}
void RoadrunnerWindow::modeCB(Widget w, XtPointer client, XtPointer) {
    ((RoadrunnerWindow*)client)->setMode(strcmp(XtName(w), "Remix") == 0 ? 1 : 0);
}
void RoadrunnerWindow::setMode(int mode) {
    _mode = mode;
    XtSetSensitive(_loadBtn, mode == 1);       // remix controls only in Remix mode
    XtSetSensitive(_strength, mode == 1);
}
void RoadrunnerWindow::exposeCB(Widget, XtPointer client, XtPointer) {
    ((RoadrunnerWindow*)client)->redraw();
}
void RoadrunnerWindow::resizeCB(Widget, XtPointer client, XtPointer) {
    ((RoadrunnerWindow*)client)->onCanvasResize();
}
void RoadrunnerWindow::canvasEH(Widget, XtPointer client, XEvent* ev, Boolean*) {
    ((RoadrunnerWindow*)client)->onCanvasEvent(ev);
}
void RoadrunnerWindow::zoomCB(Widget w, XtPointer client, XtPointer) {
    RoadrunnerWindow* self = (RoadrunnerWindow*)client;
    int cw, ch; self->canvasSize(&cw, &ch);
    const char* n = XtName(w);
    if      (strcmp(n, "Fit") == 0) self->fitToWindow();
    else if (strcmp(n, "1:1") == 0) self->zoomTo(1.0f, cw / 2, ch / 2);
    else if (strcmp(n, "Zoom-") == 0) self->zoomTo(self->_zoom * 0.8f, cw / 2, ch / 2);
    else if (strcmp(n, "Zoom+") == 0) self->zoomTo(self->_zoom * 1.25f, cw / 2, ch / 2);
}
void RoadrunnerWindow::progExposeCB(Widget, XtPointer client, XtPointer) {
    ((RoadrunnerWindow*)client)->drawProgress();
}
void RoadrunnerWindow::generateCB(Widget, XtPointer client, XtPointer) {
    ((RoadrunnerWindow*)client)->onGenerate();
}
void RoadrunnerWindow::prevCB(Widget, XtPointer client, XtPointer) {
    RoadrunnerWindow* self = (RoadrunnerWindow*)client;
    self->selectImage(self->_selected - 1);
}
void RoadrunnerWindow::nextCB(Widget, XtPointer client, XtPointer) {
    RoadrunnerWindow* self = (RoadrunnerWindow*)client;
    self->selectImage(self->_selected + 1);
}
void RoadrunnerWindow::thumbExposeCB(Widget w, XtPointer client, XtPointer) {
    RoadrunnerWindow* self = (RoadrunnerWindow*)client;
    for (int i = 0; i < MAX_BATCH; i++)
        if (self->_thumbDA[i] == w) { self->redrawThumb(i); return; }
}
void RoadrunnerWindow::thumbInputCB(Widget w, XtPointer client, XtPointer call) {
    RoadrunnerWindow* self = (RoadrunnerWindow*)client;
    XmDrawingAreaCallbackStruct* cbs = (XmDrawingAreaCallbackStruct*)call;
    if (!cbs->event || cbs->event->type != ButtonPress) return;
    for (int i = 0; i < MAX_BATCH; i++)
        if (self->_thumbDA[i] == w) { self->selectImage(i); return; }
}

// -- connect to the bridge ----------------------------------------------------
int RoadrunnerWindow::connectBridge() {
    struct hostent* he = gethostbyname(_host);
    if (!he) return -1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((unsigned short)_port);
    memcpy(&addr.sin_addr, he->h_addr, he->h_length);
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) { close(fd); return -1; }
    return fd;
}

// -- ask the bridge for its model list (LIST -> names, then EOF) --------------
int RoadrunnerWindow::queryModels(char names[][64], int maxN) {
    int fd = connectBridge();
    if (fd < 0) return 0;
    struct timeval tv; tv.tv_sec = 5; tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    if (!send_all(fd, "LIST\n", 5)) { close(fd); return 0; }
    char buf[8192]; int total = 0;
    for (;;) {
        int k = read(fd, buf + total, (int)sizeof(buf) - 1 - total);
        if (k <= 0) break;
        total += k;
        if (total >= (int)sizeof(buf) - 1) break;
    }
    close(fd);
    buf[total] = '\0';
    int count = 0;
    for (char* line = strtok(buf, "\n"); line && count < maxN; line = strtok(NULL, "\n")) {
        if (!line[0]) continue;
        strncpy(names[count], line, 63); names[count][63] = '\0'; count++;
    }
    return count;
}

// -- kick off a batch ---------------------------------------------------------
void RoadrunnerWindow::onGenerate() {
    if (_fd >= 0) return;                        // already running
    if (_mode == 1 && !_hasInit) {
        setStatus("Remix mode: load an init image first."); return;
    }
    XtSetSensitive(_generate, False);
    setStatus("Connecting to sparky...");

    int fd = connectBridge();
    if (fd < 0) {
        char m[320]; sprintf(m, "Cannot reach bridge at %s:%d", _host, _port);
        setStatus(m); XtSetSensitive(_generate, True); return;
    }

    char* prompt  = XmTextGetString(_prompt);
    char* seedTxt = XmTextFieldGetString(_seed);
    int   steps   = 0; XmScaleGetValue(_steps, &steps);
    long  seed    = -1;
    if (seedTxt && seedTxt[0]) seed = strtol(seedTxt, NULL, 10);
    if (prompt) for (char* p = prompt; *p; p++) if (*p == '\n' || *p == '\r') *p = ' ';

    const char* modelTok =
        (_model[0] && strcmp(_model, "(default)") != 0) ? _model : "-";
    const int remix = (_mode == 1 && _hasInit);
    int cfgv = 10; XmScaleGetValue(_cfg, &cfgv);         // 10..30 => 1.0..3.0 (10 = off)
    int varv = 0; XmScaleGetValue(_var, &varv);          // 0..100 %
    long seedVar = (varv > 0) ? (long)(rand() & 0x7fffffff) : 0;
    int cfg100 = cfgv * 10;                              // 10..30 -> 100..300
    char* negTxt = XmTextFieldGetString(_negative);
    if (negTxt) for (char* q = negTxt; *q; q++) if (*q == '\n' || *q == '\r') *q = ' ';

    char header[320];
    if (remix) {
        int st = 60; XmScaleGetValue(_strength, &st);
        sprintf(header, "REMIX %d %d %ld %d %d %d %d %d %ld %d %s\n",
                _res, steps, seed, _count, st, _initDim, _initDim,
                cfg100, seedVar, varv, modelTok);
    } else {
        sprintf(header, "GEN %d %d %ld %d %d %ld %d %s\n",
                _res, steps, seed, _count, cfg100, seedVar, varv, modelTok);
    }
    int ok = send_all(fd, header, (int)strlen(header)) &&
             send_all(fd, prompt ? prompt : "", prompt ? (int)strlen(prompt) : 0) &&
             send_all(fd, "\n", 1) &&
             send_all(fd, negTxt ? negTxt : "", negTxt ? (int)strlen(negTxt) : 0) &&
             send_all(fd, "\n", 1);
    if (negTxt) XtFree(negTxt);
    if (ok && remix)                             // raw RGB payload follows the two text lines
        ok = send_all(fd, _initRGB, _initDim * _initDim * 3);

    // snapshot request params for File > Save (per-image seed arrives with each image)
    if (_lastPrompt) free(_lastPrompt);
    _lastPrompt = strdup(prompt ? prompt : "");
    strncpy(_lastModelSel, _model, sizeof(_lastModelSel) - 1);
    _lastModelSel[sizeof(_lastModelSel) - 1] = '\0';
    _lastRes = _res; _lastSteps = steps;

    if (prompt)  XtFree(prompt);
    if (seedTxt) XtFree(seedTxt);
    if (!ok) { setStatus("Send failed."); close(fd); XtSetSensitive(_generate, True); return; }
    armReceive(fd);
}

// Clear the batch, reset progress, and hand the socket to the Xt event loop so
// onInput() drains the tagged reply stream. Shared by Generate and Outpaint.
void RoadrunnerWindow::armReceive(int fd) {
    clearBatch();
    _progPermille = 0; drawProgress();
    fcntl(fd, F_SETFL, O_NONBLOCK);
    _fd = fd;
    _rxLen = 0; _rxHead = 0;
    setStatus("Generating on sparky...");
    XtAppContext ctx = XtWidgetToApplicationContext(_canvas);
    _inputId = XtAppAddInput(ctx, fd, (XtPointer)XtInputReadMask,
                             &RoadrunnerWindow::inputCB, (XtPointer)this);
}

// Outpaint the currently-shown image: send it as the OUTPAINT source; the bridge
// composites the zoom-out canvas + border mask and inpaints the new border.
void RoadrunnerWindow::outpaintCB(Widget, XtPointer client, XtPointer) {
    ((RoadrunnerWindow*)client)->onOutpaint();
}

void RoadrunnerWindow::onOutpaint() {
    if (_fd >= 0) return;
    if (!_curRGB) { setStatus("Outpaint: generate or select an image first."); return; }
    XtSetSensitive(_generate, False);
    setStatus("Connecting to sparky...");
    int fd = connectBridge();
    if (fd < 0) { setStatus("Cannot reach the bridge."); XtSetSensitive(_generate, True); return; }

    int steps = 0; XmScaleGetValue(_steps, &steps);
    if (steps < 12) steps = 12;                          // outpaint likes a few more
    long seed = -1; char* seedTxt = XmTextFieldGetString(_seed);
    if (seedTxt && seedTxt[0]) seed = strtol(seedTxt, NULL, 10);
    if (seedTxt) XtFree(seedTxt);
    int cfgv = 10; XmScaleGetValue(_cfg, &cfgv);
    int varv = 0;  XmScaleGetValue(_var, &varv);
    long seedVar = (varv > 0) ? (long)(rand() & 0x7fffffff) : 0;
    int cfg100 = cfgv * 10;
    const char* modelTok = (_model[0] && strcmp(_model, "(default)") != 0) ? _model : "-";

    char header[320];   // OUTPAINT reuses the REMIX field layout; strength forced by the bridge
    sprintf(header, "OUTPAINT %d %d %ld %d %d %d %d %d %ld %d %s\n",
            _res, steps, seed, 1, 90, _curW, _curH, cfg100, seedVar, varv, modelTok);
    char* prompt  = XmTextGetString(_prompt);
    char* negTxt  = XmTextFieldGetString(_negative);
    if (prompt) for (char* q = prompt; *q; q++) if (*q == '\n' || *q == '\r') *q = ' ';
    if (negTxt) for (char* q = negTxt; *q; q++) if (*q == '\n' || *q == '\r') *q = ' ';
    int ok = send_all(fd, header, (int)strlen(header)) &&
             send_all(fd, prompt ? prompt : "", prompt ? (int)strlen(prompt) : 0) &&
             send_all(fd, "\n", 1) &&
             send_all(fd, negTxt ? negTxt : "", negTxt ? (int)strlen(negTxt) : 0) &&
             send_all(fd, "\n", 1) &&
             send_all(fd, _curRGB, (long)_curW * _curH * 3);

    if (_lastPrompt) free(_lastPrompt);
    _lastPrompt = strdup(prompt ? prompt : "");
    strncpy(_lastModelSel, _model, sizeof(_lastModelSel) - 1);
    _lastModelSel[sizeof(_lastModelSel) - 1] = '\0';
    _lastRes = _res; _lastSteps = steps;
    if (prompt) XtFree(prompt);
    if (negTxt) XtFree(negTxt);
    if (!ok) { setStatus("Send failed."); close(fd); XtSetSensitive(_generate, True); return; }
    armReceive(fd);
}

// -- grow the receive buffer by n bytes; 0 on OOM -----------------------------
int RoadrunnerWindow::appendRx(const unsigned char* d, int n) {
    if (_rxLen + n > _rxCap) {
        int cap = _rxCap ? _rxCap : 65536;
        while (cap < _rxLen + n) cap *= 2;
        unsigned char* p = (unsigned char*)realloc(_rx, cap);
        if (!p) return 0;
        _rx = p; _rxCap = cap;
    }
    memcpy(_rx + _rxLen, d, n);
    _rxLen += n;
    return 1;
}

// -- end the batch, re-arm the UI ---------------------------------------------
void RoadrunnerWindow::finishBatch(const char* status) {
    if (_inputId) { XtRemoveInput(_inputId); _inputId = 0; }
    if (_fd >= 0) { close(_fd); _fd = -1; }
    _rxLen = 0; _rxHead = 0;                     // keep _rx allocated for reuse
    setStatus(status);
    XtSetSensitive(_generate, True);
}

void RoadrunnerWindow::inputCB(XtPointer client, int*, XtInputId*) {
    ((RoadrunnerWindow*)client)->onInput();
}

// Drain the socket, then parse as many whole tagged messages as have arrived.
// All multi-byte fields are memcpy'd out before ntohl — MIPS faults on unaligned
// word loads, so we must not cast into the middle of _rx.
void RoadrunnerWindow::onInput() {
    unsigned char chunk[65536];
    int eof = 0;
    for (;;) {
        int n = read(_fd, chunk, sizeof(chunk));
        if (n > 0) { if (!appendRx(chunk, n)) { finishBatch("Out of memory."); return; } continue; }
        if (n == 0) { eof = 1; break; }
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        if (errno == EINTR) continue;
        finishBatch("Read error."); return;
    }

    for (;;) {
        int avail = _rxLen - _rxHead;
        if (avail < 8) break;
        unsigned char* m = _rx + _rxHead;
        if (memcmp(m, "F2K1", 4) != 0) { finishBatch("Bad message (magic)."); return; }
        unsigned int type; memcpy(&type, m + 4, 4); type = ntohl(type);

        if (type == MSG_PROGRESS) {
            if (avail < 24) break;
            unsigned int len; memcpy(&len, m + 20, 4); len = ntohl(len);
            long need = 24 + (long)len;
            if (avail < need) break;
            unsigned int idx, total, pm;
            memcpy(&idx, m + 8, 4); memcpy(&total, m + 12, 4); memcpy(&pm, m + 16, 4);
            char phase[32]; int c = (len > 31) ? 31 : (int)len;
            memcpy(phase, m + 24, c); phase[c] = '\0';
            onProgress((int)ntohl(idx), (int)ntohl(total), (int)ntohl(pm), phase);
            _rxHead += (int)need;
        } else if (type == MSG_IMAGE) {
            if (avail < 24) break;
            unsigned int idx, nw, nh, ns;
            memcpy(&idx, m + 8, 4); memcpy(&nw, m + 12, 4);
            memcpy(&nh, m + 16, 4); memcpy(&ns, m + 20, 4);
            int w = (int)ntohl(nw), h = (int)ntohl(nh);
            long need = 24 + (long)w * h * 3;
            if (avail < need) break;
            addBatchImage((int)ntohl(idx), w, h, (unsigned long)ntohl(ns), m + 24);
            _rxHead += (int)need;
        } else if (type == MSG_DONE) {
            if (avail < 12) break;
            _rxHead += 12;
            char s[96]; sprintf(s, "Batch complete: %d image%s.", _batchN, _batchN == 1 ? "" : "s");
            finishBatch(s); return;
        } else if (type == MSG_ERROR) {
            if (avail < 12) break;
            unsigned int len; memcpy(&len, m + 8, 4); len = ntohl(len);
            if (avail < 12 + (long)len) break;
            char msg[512]; int c = (len > 511) ? 511 : (int)len;
            memcpy(msg, m + 12, c); msg[c] = '\0';
            char e[600]; sprintf(e, "Bridge error: %s", msg);
            finishBatch(e); return;
        } else { finishBatch("Unknown message type."); return; }
    }

    // compact consumed bytes to the front so _rx doesn't grow unbounded
    if (_rxHead > 0) {
        int rem = _rxLen - _rxHead;
        if (rem > 0) memmove(_rx, _rx + _rxHead, rem);
        _rxLen = rem; _rxHead = 0;
    }
    if (eof) finishBatch("Connection closed early.");
}

// -- progress -----------------------------------------------------------------
void RoadrunnerWindow::onProgress(int idx, int total, int permille, const char* phase) {
    _progPermille = permille;
    drawProgress();
    char m[128];
    sprintf(m, "Image %d/%d: %s  (%d%%)", idx + 1, total, phase, permille / 10);
    setStatus(m);
}

void RoadrunnerWindow::drawProgress() {
    Window win = XtWindow(_progDA);
    if (!win) return;
    if (!_gc) _gc = XCreateGC(_dpy, win, 0, NULL);
    Dimension w = 0, h = 0;
    XtVaGetValues(_progDA, XmNwidth, &w, XmNheight, &h, NULL);
    Screen* scr = XtScreen(_progDA);
    XSetForeground(_dpy, _gc, BlackPixelOfScreen(scr));
    XFillRectangle(_dpy, win, _gc, 0, 0, w, h);
    int fw = (int)((long)w * _progPermille / 1000);
    XSetForeground(_dpy, _gc, WhitePixelOfScreen(scr));
    XFillRectangle(_dpy, win, _gc, 0, 0, fw, h);
}

// -- batch handling -----------------------------------------------------------
void RoadrunnerWindow::clearBatch() {
    for (int i = 0; i < MAX_BATCH; i++) {
        if (_batch[i].rgb)   { free(_batch[i].rgb); _batch[i].rgb = NULL; }
        if (_batch[i].thumb) { XDestroyImage(_batch[i].thumb); _batch[i].thumb = NULL; }
        if (XtIsManaged(_thumbDA[i])) XtUnmanageChild(_thumbDA[i]);
    }
    _batchN = 0; _selected = -1;
    XtSetSensitive(_prevBtn, False);
    XtSetSensitive(_nextBtn, False);
}

void RoadrunnerWindow::addBatchImage(int idx, int w, int h, unsigned long seed,
                                     const unsigned char* rgb) {
    if (idx < 0 || idx >= MAX_BATCH) return;
    long nbytes = (long)w * h * 3;
    unsigned char* copy = (unsigned char*)malloc(nbytes);
    if (!copy) { setStatus("Out of memory (image)."); return; }
    memcpy(copy, rgb, nbytes);
    if (_batch[idx].rgb)   free(_batch[idx].rgb);
    if (_batch[idx].thumb) { XDestroyImage(_batch[idx].thumb); _batch[idx].thumb = NULL; }
    _batch[idx].rgb = copy; _batch[idx].w = w; _batch[idx].h = h; _batch[idx].seed = seed;

    _batch[idx].thumb = makeScaledXImage(copy, w, h, THUMB, THUMB);
    if (idx + 1 > _batchN) _batchN = idx + 1;
    if (!XtIsManaged(_thumbDA[idx])) XtManageChild(_thumbDA[idx]);
    redrawThumb(idx);

    if (idx == 0) selectImage(0);        // show the first as soon as it lands
    else if (_selected >= 0) {           // keep nav state fresh as more arrive
        XtSetSensitive(_nextBtn, _selected < _batchN - 1);
    }
}

void RoadrunnerWindow::selectImage(int k) {
    if (k < 0 || k >= _batchN || !_batch[k].rgb) return;
    int old = _selected;
    _selected = k;
    showImage(_batch[k].w, _batch[k].h, _batch[k].rgb);
    if (old >= 0 && old < _batchN) redrawThumb(old);
    redrawThumb(k);
    XtSetSensitive(_prevBtn, _selected > 0);
    XtSetSensitive(_nextBtn, _selected < _batchN - 1);
    char m[128];
    sprintf(m, "Image %d/%d  seed %lu", k + 1, _batchN, _batch[k].seed);
    setStatus(m);
}

void RoadrunnerWindow::redrawThumb(int i) {
    if (i < 0 || i >= MAX_BATCH || !_batch[i].thumb) return;
    Window win = XtWindow(_thumbDA[i]);
    if (!win) return;
    if (!_gc) _gc = XCreateGC(_dpy, win, 0, NULL);
    XPutImage(_dpy, win, _gc, _batch[i].thumb, 0, 0, 0, 0, THUMB, THUMB);
    if (i == _selected) {                // highlight the shown one
        XSetForeground(_dpy, _gc, WhitePixelOfScreen(XtScreen(_thumbDA[i])));
        XDrawRectangle(_dpy, win, _gc, 0, 0, THUMB - 1, THUMB - 1);
        XDrawRectangle(_dpy, win, _gc, 1, 1, THUMB - 3, THUMB - 3);
    }
}

// Nearest-neighbour resample of source RGB (sw x sh) into an XImage (dw x dh)
// packed for this display's visual. Used for the main view (any zoom) and thumbs.
XImage* RoadrunnerWindow::makeScaledXImage(const unsigned char* rgb,
                                           int sw, int sh, int dw, int dh) {
    Screen*  scr    = XtScreen(_canvas);
    Visual*  visual = DefaultVisualOfScreen(scr);
    int      depth  = DefaultDepthOfScreen(scr);
    // NB: in C++ the Xlib Visual member 'class' is exposed as 'c_class'.
    if (visual->c_class != TrueColor && visual->c_class != DirectColor) {
        setStatus("Unsupported X visual (need TrueColor).");
        return NULL;
    }
    XImage* img = XCreateImage(_dpy, visual, depth, ZPixmap, 0, NULL, dw, dh, 32, 0);
    if (!img) return NULL;
    img->data = (char*)malloc(img->bytes_per_line * dh);
    if (!img->data) { XDestroyImage(img); return NULL; }

    unsigned long masks[3];
    masks[0] = visual->red_mask; masks[1] = visual->green_mask; masks[2] = visual->blue_mask;
    int shift[3], bits[3];
    for (int c = 0; c < 3; c++) {
        unsigned long mm = masks[c];
        int s = 0; while (mm && !(mm & 1)) { mm >>= 1; s++; }
        int b = 0; while (mm & 1) { mm >>= 1; b++; }
        shift[c] = s; bits[c] = b;
    }
    for (int y = 0; y < dh; y++) {
        int sy = (dh == sh) ? y : (int)((long)y * sh / dh);
        for (int x = 0; x < dw; x++) {
            int sx = (dw == sw) ? x : (int)((long)x * sw / dw);
            const unsigned char* sp = rgb + ((long)sy * sw + sx) * 3;
            unsigned long pixel = 0;
            for (int c = 0; c < 3; c++) {
                unsigned long v = (bits[c] >= 8)
                    ? ((unsigned long)sp[c] << (bits[c] - 8))
                    : ((unsigned long)sp[c] >> (8 - bits[c]));
                pixel |= (v << shift[c]) & masks[c];
            }
            XPutPixel(img, x, y, pixel);
        }
    }
    return img;
}

void RoadrunnerWindow::canvasSize(int* cw, int* ch) {
    Dimension w = 0, h = 0;
    XtVaGetValues(_canvas, XmNwidth, &w, XmNheight, &h, NULL);
    *cw = w ? w : 1; *ch = h ? h : 1;
}

// Show a new image: own a copy of its RGB, then fit it into the canvas.
void RoadrunnerWindow::showImage(int w, int h, const unsigned char* rgb) {
    long n = (long)w * h * 3;
    unsigned char* copy = (unsigned char*)malloc(n);
    if (!copy) { setStatus("Out of memory (display)."); return; }
    memcpy(copy, rgb, n);
    if (_curRGB) free(_curRGB);
    _curRGB = copy; _curW = w; _curH = h;
    _fitMode = 1;
    fitToWindow();                 // sets zoom/offset, builds _disp, redraws
}

// (Re)build the scaled display image for the current _zoom (clamped by MAX_DISP).
void RoadrunnerWindow::buildDisp() {
    if (!_curRGB) return;
    int dw = (int)(_curW * _zoom + 0.5f); if (dw < 1) dw = 1;
    int dh = (int)(_curH * _zoom + 0.5f); if (dh < 1) dh = 1;
    if (dw > MAX_DISP || dh > MAX_DISP) {      // clamp zoom so the buffer stays sane
        int longEdge = (_curW > _curH) ? _curW : _curH;
        _zoom = (float)MAX_DISP / longEdge;
        dw = (int)(_curW * _zoom + 0.5f); dh = (int)(_curH * _zoom + 0.5f);
    }
    if (_disp) { XDestroyImage(_disp); _disp = NULL; }
    _disp = makeScaledXImage(_curRGB, _curW, _curH, dw, dh);
    _dispW = dw; _dispH = dh;
}

void RoadrunnerWindow::fitToWindow() {
    if (!_curRGB) return;
    int cw, ch; canvasSize(&cw, &ch);
    float zx = (float)cw / _curW, zy = (float)ch / _curH;
    float z = (zx < zy) ? zx : zy;
    if (z > 1.0f) z = 1.0f;                     // don't upscale small images to "fit"
    if (z < MIN_ZOOM) z = MIN_ZOOM;
    _zoom = z; _fitMode = 1;
    buildDisp();
    _offX = (cw - _dispW) / 2;                  // centre
    _offY = (ch - _dispH) / 2;
    redraw();
}

void RoadrunnerWindow::zoomTo(float nz, int cx, int cy) {
    if (!_curRGB) return;
    if (nz < MIN_ZOOM) nz = MIN_ZOOM;
    if (nz > MAX_ZOOM) nz = MAX_ZOOM;
    float srcx = (cx - _offX) / _zoom;          // source point under (cx,cy)
    float srcy = (cy - _offY) / _zoom;
    _zoom = nz; _fitMode = 0;
    buildDisp();                                // may re-clamp _zoom
    _offX = cx - (int)(srcx * _zoom);           // keep that point under the cursor
    _offY = cy - (int)(srcy * _zoom);
    redraw();
}

void RoadrunnerWindow::onCanvasResize() {
    if (_fitMode) fitToWindow();
    else redraw();
}

void RoadrunnerWindow::onCanvasEvent(XEvent* ev) {
    if (ev->type == ButtonPress && ev->xbutton.button == Button1) {
        _dragging = 1;
        _dragX0 = ev->xbutton.x; _dragY0 = ev->xbutton.y;
        _offX0 = _offX; _offY0 = _offY;
    } else if (ev->type == ButtonRelease && ev->xbutton.button == Button1) {
        _dragging = 0;
    } else if (ev->type == MotionNotify && _dragging) {
        _offX = _offX0 + (ev->xmotion.x - _dragX0);
        _offY = _offY0 + (ev->xmotion.y - _dragY0);
        redraw();
    }
}

// Render the frame into an off-screen pixmap, then flip it to the window in one
// XCopyArea. The window never shows the intermediate black clear, so panning is
// flicker-free (no strobing).
void RoadrunnerWindow::redraw() {
    Window win = XtWindow(_canvas);
    if (!win) return;
    if (!_gc) _gc = XCreateGC(_dpy, win, 0, NULL);
    int cw, ch; canvasSize(&cw, &ch);

    // (re)size the back buffer to match the canvas
    if (!_buf || _bufW != cw || _bufH != ch) {
        if (_buf) XFreePixmap(_dpy, _buf);
        _buf = XCreatePixmap(_dpy, win, cw, ch, DefaultDepthOfScreen(XtScreen(_canvas)));
        _bufW = cw; _bufH = ch;
    }

    // paint into the buffer: black, then the visible portion of the scaled image
    XSetForeground(_dpy, _gc, BlackPixelOfScreen(XtScreen(_canvas)));
    XFillRectangle(_dpy, _buf, _gc, 0, 0, cw, ch);
    if (_disp) {
        int sx = (_offX < 0) ? -_offX : 0;
        int sy = (_offY < 0) ? -_offY : 0;
        int dx = (_offX > 0) ? _offX : 0;
        int dy = (_offY > 0) ? _offY : 0;
        int ww = _dispW - sx; if (ww > cw - dx) ww = cw - dx;
        int hh = _dispH - sy; if (hh > ch - dy) hh = ch - dy;
        if (ww > 0 && hh > 0)
            XPutImage(_dpy, _buf, _gc, _disp, sx, sy, dx, dy, ww, hh);
    }
    XCopyArea(_dpy, _buf, win, _gc, 0, 0, cw, ch, 0, 0);   // atomic flip
}

// -- File menu ----------------------------------------------------------------
void RoadrunnerWindow::quitCB(Widget, XtPointer, XtPointer) { exit(0); }

void RoadrunnerWindow::saveCB(Widget, XtPointer client, XtPointer) {
    ((RoadrunnerWindow*)client)->onSave();
}

void RoadrunnerWindow::onSave() {
    if (_selected < 0) { setStatus("Nothing to save yet — generate an image first."); return; }
    Widget dlg = XmCreateFileSelectionDialog(mainWindowWidget(), (char*)"saveDlg", NULL, 0);
    char sug[128]; sprintf(sug, "roadrunner_%lu.png", _batch[_selected].seed);
    XmString s = XmStringCreateLocalized(sug);
    XtVaSetValues(dlg, XmNdirSpec, s, NULL);
    XmStringFree(s);
    XtAddCallback(dlg, XmNokCallback,     &RoadrunnerWindow::saveOkCB,     (XtPointer)this);
    XtAddCallback(dlg, XmNcancelCallback, &RoadrunnerWindow::saveCancelCB, (XtPointer)this);
    Widget help = XmFileSelectionBoxGetChild(dlg, XmDIALOG_HELP_BUTTON);
    if (help) XtUnmanageChild(help);
    XtManageChild(dlg);
}

void RoadrunnerWindow::saveCancelCB(Widget w, XtPointer, XtPointer) {
    XtDestroyWidget(XtParent(w));
}

void RoadrunnerWindow::saveOkCB(Widget w, XtPointer client, XtPointer call) {
    RoadrunnerWindow* self = (RoadrunnerWindow*)client;
    XmFileSelectionBoxCallbackStruct* cbs = (XmFileSelectionBoxCallbackStruct*)call;
    char* path = NULL;
    XmStringGetLtoR(cbs->value, XmFONTLIST_DEFAULT_TAG, &path);
    if (path) { self->doSave(path); XtFree(path); }
    XtDestroyWidget(XtParent(w));
}

// Write <base>.png (via stb) + <base>.txt (prompt + params) for the shown image.
void RoadrunnerWindow::doSave(const char* base0) {
    if (_selected < 0) { setStatus("Nothing to save."); return; }
    BatchImg& im = _batch[_selected];
    char base[1024];
    strncpy(base, base0, sizeof(base) - 1); base[sizeof(base) - 1] = '\0';
    int n = (int)strlen(base);
    if (n > 4 && (strcmp(base + n - 4, ".png") == 0 || strcmp(base + n - 4, ".txt") == 0))
        base[n - 4] = '\0';

    char png[1040], txt[1040];
    sprintf(png, "%s.png", base);
    sprintf(txt, "%s.txt", base);

    int okp = stbi_write_png(png, im.w, im.h, 3, im.rgb, im.w * 3);
    int okt = 0;
    FILE* f = fopen(txt, "w");
    if (f) {
        fprintf(f, "prompt: %s\n", _lastPrompt ? _lastPrompt : "");
        fprintf(f, "model: %s\n",
                (_lastModelSel[0] && strcmp(_lastModelSel, "(default)") != 0)
                    ? _lastModelSel : "default");
        fprintf(f, "resolution: %d\n", _lastRes);
        fprintf(f, "steps: %d\n", _lastSteps);
        fprintf(f, "seed: %lu\n", im.seed);
        fprintf(f, "size: %dx%d\n", im.w, im.h);
        fprintf(f, "bridge: %s:%d\n", _host, _port);
        fprintf(f, "generator: F2K_CUDA / FLUX.2-klein on sparky, via octane_bridge\n");
        fclose(f); okt = 1;
    }
    char m[1200];
    if (okp && okt) sprintf(m, "Saved %s + .txt", png);
    else if (okp)   sprintf(m, "Saved %s (params write failed)", png);
    else            sprintf(m, "Save FAILED for %s", png);
    setStatus(m);
}

// -- remix: load an init image ------------------------------------------------
void RoadrunnerWindow::loadCB(Widget, XtPointer client, XtPointer) {
    ((RoadrunnerWindow*)client)->onLoad();
}

void RoadrunnerWindow::onLoad() {
    Widget dlg = XmCreateFileSelectionDialog(mainWindowWidget(), (char*)"loadDlg", NULL, 0);
    XtAddCallback(dlg, XmNokCallback,     &RoadrunnerWindow::loadOkCB,     (XtPointer)this);
    XtAddCallback(dlg, XmNcancelCallback, &RoadrunnerWindow::saveCancelCB, (XtPointer)this);
    Widget help = XmFileSelectionBoxGetChild(dlg, XmDIALOG_HELP_BUTTON);
    if (help) XtUnmanageChild(help);
    XtManageChild(dlg);
}

void RoadrunnerWindow::loadOkCB(Widget w, XtPointer client, XtPointer call) {
    RoadrunnerWindow* self = (RoadrunnerWindow*)client;
    XmFileSelectionBoxCallbackStruct* cbs = (XmFileSelectionBoxCallbackStruct*)call;
    char* path = NULL;
    XmStringGetLtoR(cbs->value, XmFONTLIST_DEFAULT_TAG, &path);
    if (path) { self->doLoad(path); XtFree(path); }
    XtDestroyWidget(XtParent(w));
}

// Decode a file (PNG/JPEG/BMP/... via stb), centre-crop to square, cap the edge,
// keep the RGB for the next Remix request, and show it in the canvas as a preview.
void RoadrunnerWindow::doLoad(const char* path) {
    int w = 0, h = 0, n = 0;
    unsigned char* px = stbi_load(path, &w, &h, &n, 3);
    if (!px) {
        char m[600]; sprintf(m, "Could not load %s (%s)", path, stbi_failure_reason());
        setStatus(m); return;
    }
    int s = (w < h) ? w : h;                    // centre-crop square
    int ox = (w - s) / 2, oy = (h - s) / 2;
    int dim = (s > INIT_MAX) ? INIT_MAX : s;    // cap what we ship
    unsigned char* out = (unsigned char*)malloc((long)dim * dim * 3);
    if (!out) { stbi_image_free(px); setStatus("Out of memory (init)."); return; }
    for (int y = 0; y < dim; y++) {
        int sy = oy + (int)((long)y * s / dim);
        for (int x = 0; x < dim; x++) {
            int sx = ox + (int)((long)x * s / dim);
            const unsigned char* sp = px + ((long)sy * w + sx) * 3;
            unsigned char* dp = out + ((long)y * dim + x) * 3;
            dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
        }
    }
    stbi_image_free(px);

    if (_initRGB) free(_initRGB);
    _initRGB = out; _initDim = dim; _hasInit = 1;
    showImage(dim, dim, out);                   // preview the init in the canvas
    char m[256];
    sprintf(m, "Loaded init %dx%d (cropped to %d). Set strength + Generate to remix.", w, h, dim);
    setStatus(m);
}

// ============================================================================
// Pull "-host X" / "-port N" out of argv before ViewKit parses the rest.
// Env vars win as the default. Anything left is handed to VkApp.
static void extractArgs(int* argc, char** argv, const char** host, int* port) {
    const char* eh = getenv("F2K_BRIDGE_HOST");
    const char* ep = getenv("F2K_BRIDGE_PORT");
    if (eh) *host = eh;
    if (ep) *port = atoi(ep);
    int out = 1;
    for (int i = 1; i < *argc; i++) {
        if (strcmp(argv[i], "-host") == 0 && i + 1 < *argc) { *host = argv[++i]; }
        else if (strcmp(argv[i], "-port") == 0 && i + 1 < *argc) { *port = atoi(argv[++i]); }
        else argv[out++] = argv[i];
    }
    *argc = out;
    argv[out] = NULL;
}

int main(int argc, char** argv) {
    const char* host = DEFAULT_HOST;
    int         port = DEFAULT_PORT;
    extractArgs(&argc, argv, &host, &port);
    srand((unsigned)time(NULL));                 // variation seeds

    VkApp* app = new VkApp((char*)"Roadrunner", &argc, argv);
    RoadrunnerWindow* win = new RoadrunnerWindow("roadrunner", host, port);
    win->show();
    app->run();
    return 0;   // not reached
}
