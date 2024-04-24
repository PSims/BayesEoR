CC = gcc
CFLAGS = -fPIC -O3 -Wall -Wextra -I${CONDA_PREFIX}/include/ -I${CONDA_PREFIX}/targets/x86_64-linux/include/
LDFLAGS = -shared -L${CONDA_PREFIX}/lib/ -L${CONDA_PREFIX}/targets/x86_64-linux/lib

TARGET = wrapmzpotrf.so

OBJECTS = wrapmzpotrf.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	    $(CC) $(LDFLAGS) -o $@ $^ -lmagma

%.o: %.c
	    $(CC) $(CFLAGS) -c -o $@ $<

clean:
	    rm -f $(TARGET) $(OBJECTS)
