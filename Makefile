HIP_PATH?= $(wildcard /opt/rocm)

HIP_PLATFORM = $(shell $(HIP_PATH)/bin/hipconfig --platform)

HIP_INCLUDE = -I${HIP_PATH}/../include

BUILD_DIR ?= build

HIPCC = $(HIP_PATH)/bin/hipcc
CPPFLAGS = -O3
LDFLAGS = -lm -lpthread

ifeq (${HIP_PLATFORM}, nvcc)
    CPPFLAGS += -arch=compute_20
endif

SRC = $(wildcard *.cpp)
OBJ = $(addprefix ${BUILD_DIR}/,$(subst .cpp,.o, $(SRC)))
BIN = ${BUILD_DIR}/farmhash-tester

.PHONY: all clean run itburn

all: ${BIN}

${BIN}: ${OBJ}
	${HIPCC} ${LDFLAGS} -o ${BIN} ${OBJ}

${BUILD_DIR}/%.o: %.cpp Makefile
	mkdir -p ${BUILD_DIR}
	${HIPCC} ${HIP_INCLUDE} ${CPPFLAGS} -c -o $@ $<  

run: itburn
itburn:
	HCC_LAZYINIT=ON ${BIN}

clean:
	rm -rf ${BUILD_DIR}
