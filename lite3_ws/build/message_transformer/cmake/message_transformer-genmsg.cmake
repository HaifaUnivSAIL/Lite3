# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "message_transformer: 2 messages, 0 services")

set(MSG_I_FLAGS "-Imessage_transformer:/lite3_ws/src/message_transformer/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(message_transformer_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/lite3_ws/src/message_transformer/msg/SimpleCMD.msg" NAME_WE)
add_custom_target(_message_transformer_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "message_transformer" "/lite3_ws/src/message_transformer/msg/SimpleCMD.msg" ""
)

get_filename_component(_filename "/lite3_ws/src/message_transformer/msg/ComplexCMD.msg" NAME_WE)
add_custom_target(_message_transformer_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "message_transformer" "/lite3_ws/src/message_transformer/msg/ComplexCMD.msg" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(message_transformer
  "/lite3_ws/src/message_transformer/msg/SimpleCMD.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/message_transformer
)
_generate_msg_cpp(message_transformer
  "/lite3_ws/src/message_transformer/msg/ComplexCMD.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/message_transformer
)

### Generating Services

### Generating Module File
_generate_module_cpp(message_transformer
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/message_transformer
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(message_transformer_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(message_transformer_generate_messages message_transformer_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/lite3_ws/src/message_transformer/msg/SimpleCMD.msg" NAME_WE)
add_dependencies(message_transformer_generate_messages_cpp _message_transformer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/lite3_ws/src/message_transformer/msg/ComplexCMD.msg" NAME_WE)
add_dependencies(message_transformer_generate_messages_cpp _message_transformer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(message_transformer_gencpp)
add_dependencies(message_transformer_gencpp message_transformer_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS message_transformer_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(message_transformer
  "/lite3_ws/src/message_transformer/msg/SimpleCMD.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/message_transformer
)
_generate_msg_eus(message_transformer
  "/lite3_ws/src/message_transformer/msg/ComplexCMD.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/message_transformer
)

### Generating Services

### Generating Module File
_generate_module_eus(message_transformer
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/message_transformer
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(message_transformer_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(message_transformer_generate_messages message_transformer_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/lite3_ws/src/message_transformer/msg/SimpleCMD.msg" NAME_WE)
add_dependencies(message_transformer_generate_messages_eus _message_transformer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/lite3_ws/src/message_transformer/msg/ComplexCMD.msg" NAME_WE)
add_dependencies(message_transformer_generate_messages_eus _message_transformer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(message_transformer_geneus)
add_dependencies(message_transformer_geneus message_transformer_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS message_transformer_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(message_transformer
  "/lite3_ws/src/message_transformer/msg/SimpleCMD.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/message_transformer
)
_generate_msg_lisp(message_transformer
  "/lite3_ws/src/message_transformer/msg/ComplexCMD.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/message_transformer
)

### Generating Services

### Generating Module File
_generate_module_lisp(message_transformer
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/message_transformer
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(message_transformer_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(message_transformer_generate_messages message_transformer_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/lite3_ws/src/message_transformer/msg/SimpleCMD.msg" NAME_WE)
add_dependencies(message_transformer_generate_messages_lisp _message_transformer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/lite3_ws/src/message_transformer/msg/ComplexCMD.msg" NAME_WE)
add_dependencies(message_transformer_generate_messages_lisp _message_transformer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(message_transformer_genlisp)
add_dependencies(message_transformer_genlisp message_transformer_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS message_transformer_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(message_transformer
  "/lite3_ws/src/message_transformer/msg/SimpleCMD.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/message_transformer
)
_generate_msg_nodejs(message_transformer
  "/lite3_ws/src/message_transformer/msg/ComplexCMD.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/message_transformer
)

### Generating Services

### Generating Module File
_generate_module_nodejs(message_transformer
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/message_transformer
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(message_transformer_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(message_transformer_generate_messages message_transformer_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/lite3_ws/src/message_transformer/msg/SimpleCMD.msg" NAME_WE)
add_dependencies(message_transformer_generate_messages_nodejs _message_transformer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/lite3_ws/src/message_transformer/msg/ComplexCMD.msg" NAME_WE)
add_dependencies(message_transformer_generate_messages_nodejs _message_transformer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(message_transformer_gennodejs)
add_dependencies(message_transformer_gennodejs message_transformer_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS message_transformer_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(message_transformer
  "/lite3_ws/src/message_transformer/msg/SimpleCMD.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/message_transformer
)
_generate_msg_py(message_transformer
  "/lite3_ws/src/message_transformer/msg/ComplexCMD.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/message_transformer
)

### Generating Services

### Generating Module File
_generate_module_py(message_transformer
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/message_transformer
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(message_transformer_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(message_transformer_generate_messages message_transformer_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/lite3_ws/src/message_transformer/msg/SimpleCMD.msg" NAME_WE)
add_dependencies(message_transformer_generate_messages_py _message_transformer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/lite3_ws/src/message_transformer/msg/ComplexCMD.msg" NAME_WE)
add_dependencies(message_transformer_generate_messages_py _message_transformer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(message_transformer_genpy)
add_dependencies(message_transformer_genpy message_transformer_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS message_transformer_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/message_transformer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/message_transformer
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(message_transformer_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/message_transformer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/message_transformer
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(message_transformer_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/message_transformer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/message_transformer
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(message_transformer_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/message_transformer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/message_transformer
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(message_transformer_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/message_transformer)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/message_transformer\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/message_transformer
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(message_transformer_generate_messages_py std_msgs_generate_messages_py)
endif()
