
GCC=g++
CLANG=clang++

all: DEBUG= -DNDEBUG
all: mtg2_trainer mtg2_parser


alld: mtg2_trainer mtg2_parser

BUILD_DIR=../bin

clean:
	rm -f *.o *.oo $(BUILD_DIR)/mtg2_trainer $(BUILD_DIR)/mtg2_parser


debug: DEBUG = -g -DDEBUG
debug: alld

wstring: WSTR = -DWSTRING
wstring: all


debug_wstring: WSTR = -DWSTRING
debug_wstring: DEBUG = -g -DDEBUG
debug_wstring: alld

OBJ_FILES_G=tss_beam.o  utils.o  tree.o  treebank.o  str_utils.o  grammar.o  parser.o  transition_system.o  features.o  classifier.o  hash_utils.o  layers.o  logger.o  random_utils.o
OBJ_FILES_C=tss_beam.oo utils.oo tree.oo treebank.oo str_utils.oo grammar.oo parser.oo transition_system.oo features.oo classifier.oo hash_utils.oo layers.oo logger.oo random_utils.oo

FLAGS_GCC=-std=c++11 -O3 -Wall -Wno-sign-compare $(DEBUG) $(WSTR) -I../lib

mtg2_trainer: $(OBJ_FILES_G) mtg.cpp
	$(GCC)       $(FLAGS_GCC)   $(OBJ_FILES_G)   mtg.cpp   -o $(BUILD_DIR)/mtg2_trainer

mtg2_parser: $(OBJ_FILES_G) mtg_parse.cpp
	$(GCC)       $(FLAGS_GCC)   $(OBJ_FILES_G)   mtg_parse.cpp   -o $(BUILD_DIR)/mtg2_parser


%.o: %.cpp %.h
	$(GCC)       $(FLAGS_GCC)    -o $@ -c $<

%.oo: %.cpp %.h
	$(CLANG)     $(FLAGS_CLA)    -o $@ -c $<
