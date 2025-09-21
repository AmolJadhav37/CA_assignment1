CXX = g++
CXXFLAGS = -std=c++17 -O2

SOURCES := $(wildcard *.cpp)
TARGETS := $(SOURCES:.cpp=.out)

all: $(TARGETS)

%.out: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
