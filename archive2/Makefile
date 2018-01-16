# Compiler and standard.
CC = gcc -std=c99

# Project name.
PROJ = bc

# Source files.
SRCS = main.c
SRCS+=

# Warnings flags.
CFLAGS = -Wshadow -Wall -Wpedantic -Wextra

# Debugging flags.
CFLAGS+= -g

# Optimization flags.
CFLAGS+= -O3 -march=native

# Linker flags.
LDFLAGS = -lm -ltensorflow

# Linker.
$(PROJ): $(SRCS:.c=.o)
	$(CC) $(CFLAGS) $(SRCS:.c=.o) $(LDFLAGS) -o $(PROJ)

# Compiler template; generates dependency targets.
%.o : %.c
	$(CC) $(CFLAGS) -MMD -MP -MT $@ -MF $*.td -c $<
	@mv -f $*.td $*.d

# All dependency targets.
%.d: ;
-include *.d

clean:
	rm -f vgcore.*
	rm -f cachegrind.out.*
	rm -f callgrind.out.*
	rm -f $(PROJ)
	rm -f $(SRCS:.c=.o)
	rm -f $(SRCS:.c=.d)
