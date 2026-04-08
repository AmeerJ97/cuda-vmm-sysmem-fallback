CC = gcc
NVCC = nvcc
CFLAGS = -Wall -Wextra -O2 -fPIC -pthread
LDFLAGS = -shared -ldl -pthread

TARGET = libcuda_vmm_fallback.so
SRC = src/shim.c
PREFIX = /usr/local

.PHONY: all clean install uninstall test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

install: $(TARGET)
	install -Dm755 $(TARGET) $(PREFIX)/lib/$(TARGET)

uninstall:
	rm -f $(PREFIX)/lib/$(TARGET)

clean:
	rm -f $(TARGET) tests/test_alloc

test: $(TARGET)
	$(NVCC) -o tests/test_alloc tests/test_alloc.cu -lcuda
	LD_PRELOAD=./$(TARGET) tests/test_alloc
	@echo "All tests passed."
