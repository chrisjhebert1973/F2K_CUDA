// Roadrunner — an SGI Octane (IRIX / MIPSpro / ViewKit) front-end for the
// F2K_CUDA image generator running on the DGX Spark ("sparky").
//
// A ViewKit/Motif app: type a prompt, pick resolution / steps / seed, hit
// Generate. The request goes over the LAN to octane_bridge on the Spark, which
// drives the resident CUDA worker and streams back raw RGB. We wrap that RGB in
// an XImage and blit it into an XmDrawingArea. No JSON, no image libraries, no
// OpenGL on this side — just Xlib.
//
//   build:  make            (see Makefile — MIPSpro CC + ViewKit + Motif)
//   run:    ./roadrunner -host sparky -port 1974
//           (or set F2K_BRIDGE_HOST / F2K_BRIDGE_PORT)
//
// ---- wire protocol (must match tools/octane_bridge.cpp) --------------------
// Request  (ASCII, two newline-terminated lines):
//     "GEN <res> <steps> <seed> <model>\n"   ints; seed < 0 => bridge randomises;
//                                            model token, "-" => bridge default
//     "<prompt>\n"
//   ("LIST\n" instead returns newline-separated model names, then EOF.)
// Response (binary; every int32 in network byte order — this box is big-endian
// MIPS, the Spark is little-endian, so ntohl earns its keep):
//     'F','2','K','1' | int32 status
//        ok:  int32 w, int32 h, int32 seed, w*h*3 RGB bytes
//        err: int32 msglen, msglen bytes

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

// Real PNG output with no system libs (stb rolls its own deflate). Bundled in
// this directory so the Octane build is self-contained. If MIPSpro's C++ front
// end fights the header, compile it instead as C into a stbiw.o (see README).
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// -- config -------------------------------------------------------------------
static const char* DEFAULT_HOST = "sparky";
static const int   DEFAULT_PORT = 1974;
static const int   RES_CHOICES[] = { 256, 512, 768, 1024 };
static const int   N_RES = 4;

// ============================================================================
class RoadrunnerWindow : public VkSimpleWindow {
public:
    RoadrunnerWindow(const char* name, const char* host, int port);
    virtual ~RoadrunnerWindow();
    virtual const char* className() { return "RoadrunnerWindow"; }

private:
    // widgets
    Widget _prompt;                 // scrolled XmText (multi-line)
    Widget _seed;                   // XmTextField
    Widget _steps;                  // XmScale
    Widget _resToggles[N_RES];      // XmToggleButtons in a radio box
    Widget _generate;               // XmPushButton
    Widget _status;                 // XmLabel
    Widget _canvas;                 // XmDrawingArea

    // network target
    char   _host[256];
    int    _port;
    int    _res;                    // current resolution selection
    char   _model[128];             // current model selection ("(default)" => "-")

    // image state
    Display* _dpy;
    XImage*  _image;                // current picture (owns its data)
    GC       _gc;

    // last-generation snapshot (for File > Save)
    unsigned char* _lastRGB;        // copy of the last image's raw RGB
    int            _lastW, _lastH;
    unsigned long  _lastSeed;       // seed the bridge actually used (echoed back)
    char*          _lastPrompt;     // strdup of the prompt sent
    char           _lastModelSel[128];
    int            _lastRes, _lastSteps;

    // -- callbacks (static trampolines -> instance methods) --
    static void generateCB(Widget, XtPointer client, XtPointer);
    static void resCB(Widget, XtPointer client, XtPointer);
    static void exposeCB(Widget, XtPointer client, XtPointer);
    static void inputCB(XtPointer client, int* source, XtInputId* id);
    static void modelCB(Widget, XtPointer client, XtPointer);
    static void saveCB(Widget, XtPointer client, XtPointer);
    static void saveOkCB(Widget, XtPointer client, XtPointer);
    static void saveCancelCB(Widget, XtPointer client, XtPointer);
    static void quitCB(Widget, XtPointer client, XtPointer);

    void onGenerate();
    void onExpose();
    void onInput();                                  // socket became readable
    void onSave();                                   // File > Save: pop the dialog
    void doSave(const char* base);                   // write <base>.png + .txt

    // -- helpers --
    void   setStatus(const char* msg);
    int    connectBridge();                          // -> fd or -1
    int    queryModels(char names[][64], int maxN);  // LIST query -> count
    int    appendRx(const unsigned char* d, int n);  // grow _rx; 0 on OOM
    void   finishGen(const char* status);            // tear down the receive
    void   showImage(int w, int h, const unsigned char* rgb);
    void   redraw();

    // -- async receive state (XtAppAddInput-driven) --
    int        _fd;         // active socket, -1 when idle
    XtInputId  _inputId;    // Xt input source, 0 when none registered
    unsigned char* _rx;     // growing receive buffer (kept + reused across runs)
    int        _rxLen;      // bytes accumulated so far this run
    int        _rxCap;      // allocated capacity of _rx
    int        _rxStatus;   // parsed response status word (0 = ok)
    int        _haveHead;   // 1 once magic+status have been parsed
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
    _image = NULL;
    _gc    = NULL;
    _dpy   = NULL;
    _fd = -1; _inputId = 0;
    _rx = NULL; _rxLen = 0; _rxCap = 0;
    _rxStatus = 0; _haveHead = 0;
    _model[0] = '\0';
    _lastRGB = NULL; _lastW = 0; _lastH = 0; _lastSeed = 0;
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
    XtVaCreateManagedWidget("sep", xmSeparatorWidgetClass, filePD, NULL);
    Widget quitItem = XtVaCreateManagedWidget("Quit", xmPushButtonWidgetClass, filePD, NULL);
    XtAddCallback(quitItem, XmNactivateCallback, &RoadrunnerWindow::quitCB, (XtPointer)this);

    // Root form: a control column on the left, the image canvas filling the rest.
    Widget form = XmCreateForm(parent, (char*)"form", NULL, 0);

    // ---- control column (left) ---------------------------------------------
    Widget panel = XtVaCreateManagedWidget("panel", xmFormWidgetClass, form,
        XmNtopAttachment,    XmATTACH_FORM,
        XmNbottomAttachment, XmATTACH_FORM,
        XmNleftAttachment,   XmATTACH_FORM,
        XmNwidth,            300,
        NULL);

    Widget promptLbl = XtVaCreateManagedWidget("Prompt:", xmLabelWidgetClass, panel,
        XmNtopAttachment,  XmATTACH_FORM,
        XmNleftAttachment, XmATTACH_FORM,
        XmNalignment,      XmALIGNMENT_BEGINNING,
        NULL);

    // multi-line, editable, word-wrapped prompt box
    Arg args[12]; int n = 0;
    XtSetArg(args[n], XmNeditMode, XmMULTI_LINE_EDIT); n++;
    XtSetArg(args[n], XmNwordWrap, True);              n++;
    XtSetArg(args[n], XmNrows, 4);                     n++;
    XtSetArg(args[n], XmNcolumns, 32);                 n++;
    _prompt = XmCreateScrolledText(panel, (char*)"prompt", args, n);
    XmTextSetString(_prompt, (char*)"a black and white Akita husky dog "
                                    "sitting on a race car, cinematic");
    XtManageChild(_prompt);
    // The ScrolledText's real geometry parent is its ScrolledWindow wrapper.
    Widget promptSW = XtParent(_prompt);
    XtVaSetValues(promptSW,
        XmNtopAttachment,    XmATTACH_WIDGET, XmNtopWidget, promptLbl,
        XmNleftAttachment,   XmATTACH_FORM,
        XmNrightAttachment,  XmATTACH_FORM,
        NULL);

    // ---- model option menu (populated from the bridge's LIST query) --------
    char models[32][64];
    int nModels = queryModels(models, 32);
    if (nModels <= 0) { strcpy(models[0], "(default)"); nModels = 1; }  // bridge down
    Widget modelPD = XmCreatePulldownMenu(panel, (char*)"modelPD", NULL, 0);
    Widget firstBtn = NULL;
    for (int i = 0; i < nModels; i++) {
        Widget b = XtVaCreateManagedWidget(models[i], xmPushButtonWidgetClass,
                                           modelPD, NULL);
        XtAddCallback(b, XmNactivateCallback,
                      &RoadrunnerWindow::modelCB, (XtPointer)this);
        if (i == 0) firstBtn = b;
    }
    strncpy(_model, models[0], sizeof(_model) - 1);      // default = first entry
    _model[sizeof(_model) - 1] = '\0';
    XmString mlbl = XmStringCreateLocalized((char*)"Model:");
    Arg ma[2]; int mn = 0;
    XtSetArg(ma[mn], XmNsubMenuId, modelPD);  mn++;
    XtSetArg(ma[mn], XmNlabelString, mlbl);   mn++;
    Widget modelOM = XmCreateOptionMenu(panel, (char*)"modelOM", ma, mn);
    XmStringFree(mlbl);
    XtVaSetValues(modelOM,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, promptSW,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        NULL);
    if (firstBtn) XtVaSetValues(modelOM, XmNmenuHistory, firstBtn, NULL);
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
        XmNnumColumns,      2,
        NULL);
    for (int i = 0; i < N_RES; i++) {
        char lbl[16]; sprintf(lbl, "%d", RES_CHOICES[i]);
        _resToggles[i] = XtVaCreateManagedWidget(lbl, xmToggleButtonWidgetClass, resBox,
            XmNset, (RES_CHOICES[i] == _res) ? True : False,
            NULL);
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
        XmNminimum,         1,
        XmNmaximum,         30,
        XmNvalue,           4,
        XmNshowValue,       True,
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

    // ---- generate button ---------------------------------------------------
    _generate = XtVaCreateManagedWidget("Generate", xmPushButtonWidgetClass, panel,
        XmNtopAttachment,   XmATTACH_WIDGET, XmNtopWidget, _seed,
        XmNleftAttachment,  XmATTACH_FORM,
        XmNrightAttachment, XmATTACH_FORM,
        NULL);
    XtAddCallback(_generate, XmNactivateCallback,
                  &RoadrunnerWindow::generateCB, (XtPointer)this);

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
        XmNwidth,  512,
        XmNheight, 512,
        XmNbackground, BlackPixelOfScreen(XtScreen(form)),
        NULL);
    XtAddCallback(_canvas, XmNexposeCallback,
                  &RoadrunnerWindow::exposeCB, (XtPointer)this);

    XtManageChild(form);
    addView(form);
    // wire the menu bar into the XmMainWindow (addView already set the work area)
    XmMainWindowSetAreas(parent, menubar, NULL, NULL, NULL, form);

    _dpy = XtDisplay(form);
}

RoadrunnerWindow::~RoadrunnerWindow() {
    if (_inputId) XtRemoveInput(_inputId);
    if (_fd >= 0) close(_fd);
    if (_rx)         free(_rx);
    if (_lastRGB)    free(_lastRGB);
    if (_lastPrompt) free(_lastPrompt);
    if (_image) XDestroyImage(_image);          // frees _image->data too
    if (_gc)    XFreeGC(_dpy, _gc);
}

// -- status label -------------------------------------------------------------
void RoadrunnerWindow::setStatus(const char* msg) {
    XmString s = XmStringCreateLocalized((char*)msg);
    XtVaSetValues(_status, XmNlabelString, s, NULL);
    XmStringFree(s);
    // force the label (and any pending exposes) to paint before we block
    XmUpdateDisplay(_status);
}

// -- resolution radio callback ------------------------------------------------
void RoadrunnerWindow::resCB(Widget w, XtPointer client, XtPointer) {
    RoadrunnerWindow* self = (RoadrunnerWindow*)client;
    if (!XmToggleButtonGetState(w)) return;     // only react to the newly-set one
    for (int i = 0; i < N_RES; i++)
        if (self->_resToggles[i] == w) { self->_res = RES_CHOICES[i]; return; }
}

void RoadrunnerWindow::exposeCB(Widget, XtPointer client, XtPointer) {
    ((RoadrunnerWindow*)client)->onExpose();
}
void RoadrunnerWindow::onExpose() { redraw(); }

void RoadrunnerWindow::generateCB(Widget, XtPointer client, XtPointer) {
    ((RoadrunnerWindow*)client)->onGenerate();
}

// The option menu's push buttons are named after the models, so XtName() is the
// selection. Store it for the next request.
void RoadrunnerWindow::modelCB(Widget w, XtPointer client, XtPointer) {
    RoadrunnerWindow* self = (RoadrunnerWindow*)client;
    strncpy(self->_model, XtName(w), sizeof(self->_model) - 1);
    self->_model[sizeof(self->_model) - 1] = '\0';
}

// Ask the bridge for its model list (one name per line, read to EOF). Returns
// the count. A short recv timeout keeps a silent/absent server from hanging
// startup; on any failure returns 0 and the caller falls back to "(default)".
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
        strncpy(names[count], line, 63);
        names[count][63] = '\0';
        count++;
    }
    return count;
}

// -- File menu ----------------------------------------------------------------
void RoadrunnerWindow::quitCB(Widget, XtPointer, XtPointer) { exit(0); }

void RoadrunnerWindow::saveCB(Widget, XtPointer client, XtPointer) {
    ((RoadrunnerWindow*)client)->onSave();
}

// Pop a save dialog, pre-filled with a seed-stamped default name.
void RoadrunnerWindow::onSave() {
    if (!_lastRGB) { setStatus("Nothing to save yet — generate an image first."); return; }
    Widget dlg = XmCreateFileSelectionDialog(mainWindowWidget(), (char*)"saveDlg", NULL, 0);
    char sug[128]; sprintf(sug, "roadrunner_%lu.png", _lastSeed);
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
    XtDestroyWidget(XtParent(w));                 // destroy the dialog shell
}

void RoadrunnerWindow::saveOkCB(Widget w, XtPointer client, XtPointer call) {
    RoadrunnerWindow* self = (RoadrunnerWindow*)client;
    XmFileSelectionBoxCallbackStruct* cbs = (XmFileSelectionBoxCallbackStruct*)call;
    char* path = NULL;
    XmStringGetLtoR(cbs->value, XmFONTLIST_DEFAULT_TAG, &path);
    if (path) { self->doSave(path); XtFree(path); }
    XtDestroyWidget(XtParent(w));
}

// Write <base>.png (via stb) + <base>.txt (prompt + params). A trailing
// .png/.txt on the chosen name is stripped so both land on the same base.
void RoadrunnerWindow::doSave(const char* base0) {
    if (!_lastRGB) { setStatus("Nothing to save."); return; }
    char base[1024];
    strncpy(base, base0, sizeof(base) - 1); base[sizeof(base) - 1] = '\0';
    int n = (int)strlen(base);
    if (n > 4 && (strcmp(base + n - 4, ".png") == 0 || strcmp(base + n - 4, ".txt") == 0))
        base[n - 4] = '\0';

    char png[1040], txt[1040];
    sprintf(png, "%s.png", base);
    sprintf(txt, "%s.txt", base);

    int okp = stbi_write_png(png, _lastW, _lastH, 3, _lastRGB, _lastW * 3);

    int okt = 0;
    FILE* f = fopen(txt, "w");
    if (f) {
        fprintf(f, "prompt: %s\n", _lastPrompt ? _lastPrompt : "");
        fprintf(f, "model: %s\n",
                (_lastModelSel[0] && strcmp(_lastModelSel, "(default)") != 0)
                    ? _lastModelSel : "default");
        fprintf(f, "resolution: %d\n", _lastRes);
        fprintf(f, "steps: %d\n", _lastSteps);
        fprintf(f, "seed: %lu\n", _lastSeed);
        fprintf(f, "size: %dx%d\n", _lastW, _lastH);
        fprintf(f, "bridge: %s:%d\n", _host, _port);
        fprintf(f, "generator: F2K_CUDA / FLUX.2-klein on sparky, via octane_bridge\n");
        fclose(f);
        okt = 1;
    }

    char m[1200];
    if (okp && okt) sprintf(m, "Saved %s + .txt", png);
    else if (okp)   sprintf(m, "Saved %s (params write failed)", png);
    else            sprintf(m, "Save FAILED for %s", png);
    setStatus(m);
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
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

// -- the whole request/response round-trip (blocking) -------------------------
void RoadrunnerWindow::onGenerate() {
    XtSetSensitive(_generate, False);
    setStatus("Connecting to sparky...");

    int fd = connectBridge();
    if (fd < 0) {
        char m[320];
        sprintf(m, "Cannot reach bridge at %s:%d", _host, _port);
        setStatus(m);
        XtSetSensitive(_generate, True);
        return;
    }

    // ---- build + send the two-line request ----
    char* prompt = XmTextGetString(_prompt);
    char* seedTxt = XmTextFieldGetString(_seed);
    int   steps  = 0;
    XmScaleGetValue(_steps, &steps);
    long  seed   = -1;
    if (seedTxt && seedTxt[0]) seed = strtol(seedTxt, NULL, 10);

    // strip embedded newlines from the prompt (the protocol is line-framed)
    if (prompt) for (char* p = prompt; *p; p++) if (*p == '\n' || *p == '\r') *p = ' ';

    // "(default)" (bridge was down at startup) maps to the '-' sentinel.
    const char* modelTok =
        (_model[0] && strcmp(_model, "(default)") != 0) ? _model : "-";
    char header[192];
    sprintf(header, "GEN %d %d %ld %s\n", _res, steps, seed, modelTok);
    int ok = send_all(fd, header, (int)strlen(header)) &&
             send_all(fd, prompt ? prompt : "", prompt ? (int)strlen(prompt) : 0) &&
             send_all(fd, "\n", 1);

    // snapshot the request params so File > Save can write a params sidecar even
    // after these buffers are freed (the actual seed arrives with the image).
    if (_lastPrompt) free(_lastPrompt);
    _lastPrompt = strdup(prompt ? prompt : "");
    strncpy(_lastModelSel, _model, sizeof(_lastModelSel) - 1);
    _lastModelSel[sizeof(_lastModelSel) - 1] = '\0';
    _lastRes = _res; _lastSteps = steps;

    if (prompt)  XtFree(prompt);
    if (seedTxt) XtFree(seedTxt);
    if (!ok) {
        setStatus("Send failed.");
        close(fd);
        XtSetSensitive(_generate, True);
        return;
    }

    // Hand the socket to the Xt event loop: go non-blocking and let onInput()
    // drain the framed reply as bytes arrive. The UI stays live (redraws, no
    // double-submit — the button is greyed) for the seconds the GPU is busy.
    fcntl(fd, F_SETFL, O_NONBLOCK);
    _fd = fd;
    _rxLen = 0; _haveHead = 0; _rxStatus = 0; // fresh receive (buffer is reused)
    setStatus("Generating on sparky...");
    XtAppContext ctx = XtWidgetToApplicationContext(_canvas);
    _inputId = XtAppAddInput(ctx, fd, (XtPointer)XtInputReadMask,
                             &RoadrunnerWindow::inputCB, (XtPointer)this);
}

// -- grow the receive buffer by n bytes; returns 0 on allocation failure ------
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

// -- tear down the current receive and re-arm the UI --------------------------
void RoadrunnerWindow::finishGen(const char* status) {
    if (_inputId) { XtRemoveInput(_inputId); _inputId = 0; }
    if (_fd >= 0) { close(_fd); _fd = -1; }
    _rxLen = 0; _haveHead = 0;                 // keep _rx allocated for reuse
    setStatus(status);
    XtSetSensitive(_generate, True);
}

void RoadrunnerWindow::inputCB(XtPointer client, int*, XtInputId*) {
    ((RoadrunnerWindow*)client)->onInput();
}

// Called by Xt whenever the socket is readable. Drains what's available, then
// tries to advance a small state machine over the fixed-offset framed reply.
// All multi-byte fields are memcpy'd out before ntohl — MIPS faults on unaligned
// word loads, so we must not cast into the middle of _rx.
void RoadrunnerWindow::onInput() {
    unsigned char chunk[65536];
    int eof = 0;
    for (;;) {
        int n = read(_fd, chunk, sizeof(chunk));
        if (n > 0) {
            if (!appendRx(chunk, n)) { finishGen("Out of memory."); return; }
            continue;
        }
        if (n == 0) { eof = 1; break; }        // peer closed
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        if (errno == EINTR) continue;
        finishGen("Read error."); return;
    }

    // header: magic (4) + status (4)
    if (!_haveHead) {
        if (_rxLen < 8) { if (eof) finishGen("Connection closed early."); return; }
        if (memcmp(_rx, "F2K1", 4) != 0) { finishGen("Bad response (magic)."); return; }
        unsigned int st; memcpy(&st, _rx + 4, 4);
        _rxStatus = (int)ntohl(st);
        _haveHead = 1;
    }

    if (_rxStatus != 0) {
        // error: msglen (4) + message
        if (_rxLen < 12) { if (eof) finishGen("Connection closed early."); return; }
        unsigned int nl; memcpy(&nl, _rx + 8, 4);
        long mlen = (long)ntohl(nl);
        if (_rxLen < 12 + mlen) { if (eof) finishGen("Connection closed early."); return; }
        char msg[512]; int c = (mlen > 511) ? 511 : (int)mlen;
        memcpy(msg, _rx + 12, c); msg[c] = '\0';
        char m[600]; sprintf(m, "Bridge error: %s", msg);
        finishGen(m);
        return;
    }

    // ok: width (4) + height (4) + seed (4) + w*h*3 RGB
    if (_rxLen < 20) { if (eof) finishGen("Connection closed early."); return; }
    unsigned int nw, nh, ns;
    memcpy(&nw, _rx + 8, 4); memcpy(&nh, _rx + 12, 4); memcpy(&ns, _rx + 16, 4);
    int w = (int)ntohl(nw), h = (int)ntohl(nh);
    long need = 20 + (long)w * h * 3;
    if (_rxLen < need) { if (eof) finishGen("Connection closed early."); return; }
    unsigned char* rgb = _rx + 20;

    // snapshot for File > Save (the request-time params were saved in onGenerate)
    long nbytes = (long)w * h * 3;
    unsigned char* copy = (unsigned char*)malloc(nbytes);
    if (copy) {
        memcpy(copy, rgb, nbytes);
        if (_lastRGB) free(_lastRGB);
        _lastRGB = copy; _lastW = w; _lastH = h;
        _lastSeed = (unsigned long)ntohl(ns);
    }

    showImage(w, h, rgb);                       // copies into the XImage
    char done[128]; sprintf(done, "Done: %dx%d  seed %lu.", w, h, (unsigned long)ntohl(ns));
    finishGen(done);
}

// -- wrap raw RGB in an XImage matching this display's visual, then blit -------
void RoadrunnerWindow::showImage(int w, int h, const unsigned char* rgb) {
    Screen*  scr    = XtScreen(_canvas);
    Visual*  visual = DefaultVisualOfScreen(scr);
    int      depth  = DefaultDepthOfScreen(scr);

    // We only handle TrueColor / DirectColor visuals (any Octane running a modern
    // demo will be one of these). PseudoColor would need colormap allocation.
    // NB: in C++ the Xlib Visual member 'class' is exposed as 'c_class'.
    if (visual->c_class != TrueColor && visual->c_class != DirectColor) {
        setStatus("Unsupported X visual (need TrueColor).");
        return;
    }

    if (_image) { XDestroyImage(_image); _image = NULL; }

    XImage* img = XCreateImage(_dpy, visual, depth, ZPixmap, 0,
                               NULL, w, h, 32, 0);
    if (!img) { setStatus("XCreateImage failed."); return; }
    img->data = (char*)malloc(img->bytes_per_line * h);
    if (!img->data) { XDestroyImage(img); setStatus("Image alloc failed."); return; }

    // Precompute shift/width for each channel from the visual's RGB masks so this
    // works for 15/16/24/30-bit TrueColor without special-casing.
    unsigned long masks[3];
    masks[0] = visual->red_mask; masks[1] = visual->green_mask; masks[2] = visual->blue_mask;
    int shift[3], bits[3];
    for (int c = 0; c < 3; c++) {
        unsigned long m = masks[c];
        int s = 0; while (m && !(m & 1)) { m >>= 1; s++; }
        int b = 0; while (m & 1) { m >>= 1; b++; }
        shift[c] = s; bits[c] = b;
    }

    const unsigned char* p = rgb;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            unsigned char comp[3];
            comp[0] = p[0]; comp[1] = p[1]; comp[2] = p[2]; p += 3;
            unsigned long pixel = 0;
            for (int c = 0; c < 3; c++) {
                unsigned long v = (bits[c] >= 8)
                    ? ((unsigned long)comp[c] << (bits[c] - 8))
                    : ((unsigned long)comp[c] >> (8 - bits[c]));
                pixel |= (v << shift[c]) & masks[c];
            }
            XPutPixel(img, x, y, pixel);   // handles server byte order for us
        }
    }

    _image = img;

    // grow the canvas to the image so nothing is clipped, then paint.
    XtVaSetValues(_canvas, XmNwidth, w, XmNheight, h, NULL);
    redraw();
}

void RoadrunnerWindow::redraw() {
    if (!_image) return;
    Window win = XtWindow(_canvas);
    if (!win) return;                       // not realized yet
    if (!_gc) _gc = XCreateGC(_dpy, win, 0, NULL);
    XPutImage(_dpy, win, _gc, _image, 0, 0, 0, 0, _image->width, _image->height);
}

// ============================================================================
// Pull "-host X" / "-port N" out of argv before ViewKit parses the rest, so the
// X toolkit doesn't choke on options it doesn't recognise. Env vars win as the
// default. Anything left in argv is handed to VkApp (standard X switches etc).
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

    VkApp* app = new VkApp((char*)"Roadrunner", &argc, argv);
    RoadrunnerWindow* win = new RoadrunnerWindow("roadrunner", host, port);
    win->show();
    app->run();
    return 0;   // not reached
}
