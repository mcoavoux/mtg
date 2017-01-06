
GCC=g++
CLANG=clang++

all: DEBUG= -DNDEBUG
all: mtg_gcc mtg_cla mtg_parse_gcc mtg_parse_cla


alld: mtg_gcc mtg_cla mtg_parse_gcc mtg_parse_cla

BUILD_DIR=../bin

clean:
	rm -f *.o *.oo $(BUILD_DIR)/mtg_gcc $(BUILD_DIR)/mtg_cla

#tss_beam.h tss_beam.cpp utils.cpp utils.h tree.cpp tree.h treebank.cpp treebank.h str_utils.cpp str_utils.h

debug: DEBUG = -g -DDEBUG
debug: alld


OBJ_FILES_G=tss_beam.o  utils.o  tree.o  treebank.o  str_utils.o  grammar.o  parser.o  transition_system.o  features.o  classifier.o  hash_utils.o  layers.o  logger.o
OBJ_FILES_C=tss_beam.oo utils.oo tree.oo treebank.oo str_utils.oo grammar.oo parser.oo transition_system.oo features.oo classifier.oo hash_utils.oo layers.oo logger.oo

FLAGS_GCC=-std=c++11 -O3 -Wall -Wno-sign-compare $(DEBUG) -I../lib
FLAGS_CLA=-std=c++11 -O3 -Wall -Wno-sign-compare $(DEBUG) -I../lib

mtg_gcc: $(OBJ_FILES_G) mtg.cpp
	$(GCC)       $(FLAGS_GCC)   $(OBJ_FILES_G)   mtg.cpp   -o $(BUILD_DIR)/mtg_gcc

mtg_cla: $(OBJ_FILES_C) mtg.cpp
	$(CLANG)     $(FLAGS_CLA)   $(OBJ_FILES_C)   mtg.cpp   -o $(BUILD_DIR)/mtg_cla

mtg_parse_gcc: $(OBJ_FILES_G) mtg_parse.cpp
	$(GCC)       $(FLAGS_GCC)   $(OBJ_FILES_G)   mtg_parse.cpp   -o $(BUILD_DIR)/mtg_parse_gcc

mtg_parse_cla: $(OBJ_FILES_C) mtg_parse.cpp
	$(CLANG)     $(FLAGS_CLA)   $(OBJ_FILES_C)   mtg_parse.cpp   -o $(BUILD_DIR)/mtg_parse_cla


%.o: %.cpp %.h
	$(GCC)       $(FLAGS_GCC)    -o $@ -c $<

%.oo: %.cpp %.h
	$(CLANG)     $(FLAGS_CLA)    -o $@ -c $<
