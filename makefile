# define compiler and path of libs
#CPPFLAGS = --std=c++11 -Wall -I $(DIR_INC) -Xpreprocessor -fopenmp 
CPPFLAGS = --std=c++11 -Wall -I $(DIR_INC)
CXXFLAGS = -g -O2
#LDLIBS = -lboost_program_options -lomp
LDLIBS = -L $(DIR_LD) -Wl,-rpath,$(DIR_LD) -lboost_program_options -lcnpy
# define variable path
DIR_SRC = src
DIR_INC = include
DIR_BIN = bin
DIR_OBJ = obj
DIR_DEP = dep
DIR_TEST = test
DIR_LD = lib
DIRS = $(DIR_BIN) $(DIR_DEP) $(DIR_OBJ)
vpath %.cpp $(DIR_SRC)
vpath %.cpp $(DIR_TEST)
vpath %.h 	$(DIR_INC)
vpath %.hpp $(DIR_INC)
SRCS := $(notdir $(wildcard $(DIR_SRC)/*.cpp))
DEPS = $(SRCS:.cpp=.d)
DEPS := $(addprefix $(DIR_DEP)/, $(DEPS))
OBJS = $(SRCS:.cpp=.o)
OBJS := $(addprefix $(DIR_OBJ)/, $(OBJS))
BIN := $(DIR_BIN)/net_sim

SRCS_TEST := $(subst main.cpp,main_test.cpp,$(SRCS))
DEPS_TEST = $(SRCS_TEST:.cpp=.d)
DEPS_TEST := $(addprefix $(DIR_DEP)/, $(DEPS_TEST))
OBJS_TEST = $(SRCS_TEST:.cpp=.o)
OBJS_TEST := $(addprefix $(DIR_OBJ)/, $(OBJS_TEST))
BIN_TEST := $(DIR_BIN)/net_sim_test

$(BIN) : $(DIRS) $(OBJS)
	$(CXX) $(CPPFLAGS) -o $(BIN) $(OBJS) $(LDLIBS)

$(BIN_TEST) : $(DIRS) $(OBJS_TEST)
	$(CXX) $(CPPFLAGS) -o $(BIN_TEST) $(OBJS_TEST) $(LDLIBS)

.PHONY : test
test : $(BIN_TEST)

.PHONY : debug
debug : CPPFLAGS += -DDEBUG
debug : $(BIN)

$(DIRS) : 
	@mkdir -p $@

$(DIR_OBJ)/%.o : %.cpp
	$(CXX) -o $@ -c $(CXXFLAGS) $(CPPFLAGS) $^

ifneq ($(MAKECMDGOALS),clean)
ifeq ($(MAKECMDGOALS),test)
-include $(DEPS_TEST)
else
-include $(DEPS)
endif
endif

ifeq ($(wildcard $(DIR_DEP)),)
DEP_DIR_DEP := $(DIR_DEP)
endif

# Generate prerequisites
$(DIR_DEP)/%.d : $(DEP_DIR_DEP) %.cpp
	@echo "Making $@ ..."
	@set -e; rm -f $@; \
		$(CC) -MM $(CPPFLAGS) -I $(DIR_INC) $(filter %.cpp, $^) > $@.$$$$; \
		sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
		rm -f $@.$$$$

.PHONY : clean
clean:
	rm -rf $(DIR_OBJ) $(DIR_BIN) $(DIR_DEP)
