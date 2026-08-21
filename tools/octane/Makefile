# Roadrunner — MIPSpro build for the SGI Octane (IRIX 6.5, ViewKit + Motif).
#
# Copy this whole tools/octane/ directory to the Octane and run `make`.
# Requires: MIPSpro C++ (CC), ViewKit (-lvk), Motif (-lXm). All standard on a
# dev-configured IRIX 6.5 box.
#
#   make            build ./roadrunner
#   make run        build + launch against $F2K_BRIDGE_HOST (default 'sparky')
#   make clean

CXX      = CC
ABI      = -n32                 # -64 also fine; -n32 is the usual Octane default
CXXFLAGS = $(ABI) -O2
# ViewKit pulls in Motif/Xt/X11; -lgen for basename() etc, -lm for math.
# IRIX keeps the socket/resolver calls in libc, so no -lsocket/-lnsl (those are
# Solaris-isms) — gethostbyname links straight from libc here.
LDLIBS   = -lvk -lXm -lXt -lXext -lX11 -lgen -lm

roadrunner: roadrunner.cxx
	$(CXX) $(CXXFLAGS) -o $@ roadrunner.cxx $(LDLIBS)

run: roadrunner
	./roadrunner

clean:
	rm -f roadrunner ii_files/*.o
	-rmdir ii_files 2>/dev/null

.PHONY: run clean
