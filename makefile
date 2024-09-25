all: main.c
	gcc main.c -std=c99 -Ofast -march=native -mavx -Wall -Wpedantic
