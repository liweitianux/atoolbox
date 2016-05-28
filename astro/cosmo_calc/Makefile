##
## Makefile for `calc_distance'
##
## Junhua Gu
## last modified: August 12, 2012
##

CPP= 		g++
CPPFLAGS= 	-Wall -g

TARGET= 	cosmo_calc

all: $(TARGET)

cosmo_calc: calc_distance.o ddivid.o main.o
	$(CPP) $(CPPFLAGS) $^ -o $@

calc_distance.o: calc_distance.cc calc_distance.h
	$(CPP) $(CPPFLAGS) -c $<

ddivid.o: ddivid.cc ddivid.h
	$(CPP) $(CPPFLAGS) -c $<

main.o: main.cc ddivid.h calc_distance.h
	$(CPP) $(CPPFLAGS) -c $<

clean:
	rm -f *.o
	rm -f $(TARGET)


