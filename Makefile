CFLAGS = -O3 -std=c99 -Wall -Wextra -Wpedantic -shared -fPIC

c_lib.so: c_lib.c
	$(CC) $(CFLAGS) c_lib.c -o c_lib.so
