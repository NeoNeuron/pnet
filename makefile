# define compiler and path of libs
CPPFLAGS = --std=c++11 -Wall -I $(DIR_INC)
CXXFLAGS = -g -O2
LDLIBS = -lboost_program_options
# define variable path
DIR_SRC = src
DIR_INC = include
DIR_BIN = bin
DIR_OBJ = obj
DIR_DEP = dep
DIRS = $(DIR_BIN) $(DIR_DEP) $(DIR_OBJ)
vpath %.cpp $(DIR_SRC)
vpath %.h 	$(DIR_INC)
vpath %.hpp $(DIR_INC)
SRCS := $(notdir $(wildcard $(DIR_SRC)/*.cpp))
DEPS = $(SRCS:.cpp=.d)
DEPS := $(addprefix $(DIR_DEP)/, $(DEPS))
OBJS = $(SRCS:.cpp=.o)
OBJS := $(addprefix $(DIR_OBJ)/, $(OBJS))
BIN := $(DIR_BIN)/net_sim

.PHONY : all
all : $(BIN)

$(BIN) : $(DIRS) $(OBJS)
	$(CXX) $(CPPFLAGS) -o $(BIN) $(OBJS) $(LDLIBS)

$(DIRS) : 
	@mkdir -p $@

$(DIR_OBJ)/%.o : %.cpp
	$(CXX) -o $@ -c $(CXXFLAGS) $(CPPFLAGS) $^

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPS)
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
