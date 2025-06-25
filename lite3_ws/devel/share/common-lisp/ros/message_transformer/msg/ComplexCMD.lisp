; Auto-generated. Do not edit!


(cl:in-package message_transformer-msg)


;//! \htmlinclude ComplexCMD.msg.html

(cl:defclass <ComplexCMD> (roslisp-msg-protocol:ros-message)
  ((cmd_code
    :reader cmd_code
    :initarg :cmd_code
    :type cl:integer
    :initform 0)
   (cmd_value
    :reader cmd_value
    :initarg :cmd_value
    :type cl:integer
    :initform 0)
   (type
    :reader type
    :initarg :type
    :type cl:integer
    :initform 0)
   (data
    :reader data
    :initarg :data
    :type cl:float
    :initform 0.0))
)

(cl:defclass ComplexCMD (<ComplexCMD>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ComplexCMD>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ComplexCMD)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name message_transformer-msg:<ComplexCMD> is deprecated: use message_transformer-msg:ComplexCMD instead.")))

(cl:ensure-generic-function 'cmd_code-val :lambda-list '(m))
(cl:defmethod cmd_code-val ((m <ComplexCMD>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader message_transformer-msg:cmd_code-val is deprecated.  Use message_transformer-msg:cmd_code instead.")
  (cmd_code m))

(cl:ensure-generic-function 'cmd_value-val :lambda-list '(m))
(cl:defmethod cmd_value-val ((m <ComplexCMD>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader message_transformer-msg:cmd_value-val is deprecated.  Use message_transformer-msg:cmd_value instead.")
  (cmd_value m))

(cl:ensure-generic-function 'type-val :lambda-list '(m))
(cl:defmethod type-val ((m <ComplexCMD>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader message_transformer-msg:type-val is deprecated.  Use message_transformer-msg:type instead.")
  (type m))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <ComplexCMD>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader message_transformer-msg:data-val is deprecated.  Use message_transformer-msg:data instead.")
  (data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ComplexCMD>) ostream)
  "Serializes a message object of type '<ComplexCMD>"
  (cl:let* ((signed (cl:slot-value msg 'cmd_code)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'cmd_value)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'type)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ComplexCMD>) istream)
  "Deserializes a message object of type '<ComplexCMD>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'cmd_code) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'cmd_value) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'type) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'data) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ComplexCMD>)))
  "Returns string type for a message object of type '<ComplexCMD>"
  "message_transformer/ComplexCMD")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ComplexCMD)))
  "Returns string type for a message object of type 'ComplexCMD"
  "message_transformer/ComplexCMD")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ComplexCMD>)))
  "Returns md5sum for a message object of type '<ComplexCMD>"
  "a72ae23e40d8c196548af30d3a5c3007")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ComplexCMD)))
  "Returns md5sum for a message object of type 'ComplexCMD"
  "a72ae23e40d8c196548af30d3a5c3007")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ComplexCMD>)))
  "Returns full string definition for message of type '<ComplexCMD>"
  (cl:format cl:nil "  int32 cmd_code~%  int32 cmd_value~%  int32 type~%  float64 data~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ComplexCMD)))
  "Returns full string definition for message of type 'ComplexCMD"
  (cl:format cl:nil "  int32 cmd_code~%  int32 cmd_value~%  int32 type~%  float64 data~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ComplexCMD>))
  (cl:+ 0
     4
     4
     4
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ComplexCMD>))
  "Converts a ROS message object to a list"
  (cl:list 'ComplexCMD
    (cl:cons ':cmd_code (cmd_code msg))
    (cl:cons ':cmd_value (cmd_value msg))
    (cl:cons ':type (type msg))
    (cl:cons ':data (data msg))
))
