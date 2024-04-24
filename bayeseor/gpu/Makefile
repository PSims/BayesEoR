CC = gcc
CFLAGS = -fPIC -O3 -Wall -Wextra
LDFLAGS = -shared

TARGET = wrapmzpotrf.so

OBJECTS = wrapmzpotrf.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	    $(CC) $(LDFLAGS) -o $@ $^ -lmagma

%.o: %.c
	    $(CC) $(CFLAGS) -c -o $@ $<

clean:
	    rm -f $(TARGET) $(OBJECTS)
