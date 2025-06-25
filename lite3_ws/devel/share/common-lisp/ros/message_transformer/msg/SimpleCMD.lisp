; Auto-generated. Do not edit!


(cl:in-package message_transformer-msg)


;//! \htmlinclude SimpleCMD.msg.html

(cl:defclass <SimpleCMD> (roslisp-msg-protocol:ros-message)
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
    :initform 0))
)

(cl:defclass SimpleCMD (<SimpleCMD>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SimpleCMD>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SimpleCMD)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name message_transformer-msg:<SimpleCMD> is deprecated: use message_transformer-msg:SimpleCMD instead.")))

(cl:ensure-generic-function 'cmd_code-val :lambda-list '(m))
(cl:defmethod cmd_code-val ((m <SimpleCMD>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader message_transformer-msg:cmd_code-val is deprecated.  Use message_transformer-msg:cmd_code instead.")
  (cmd_code m))

(cl:ensure-generic-function 'cmd_value-val :lambda-list '(m))
(cl:defmethod cmd_value-val ((m <SimpleCMD>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader message_transformer-msg:cmd_value-val is deprecated.  Use message_transformer-msg:cmd_value instead.")
  (cmd_value m))

(cl:ensure-generic-function 'type-val :lambda-list '(m))
(cl:defmethod type-val ((m <SimpleCMD>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader message_transformer-msg:type-val is deprecated.  Use message_transformer-msg:type instead.")
  (type m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SimpleCMD>) ostream)
  "Serializes a message object of type '<SimpleCMD>"
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
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SimpleCMD>) istream)
  "Deserializes a message object of type '<SimpleCMD>"
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
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SimpleCMD>)))
  "Returns string type for a message object of type '<SimpleCMD>"
  "message_transformer/SimpleCMD")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SimpleCMD)))
  "Returns string type for a message object of type 'SimpleCMD"
  "message_transformer/SimpleCMD")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SimpleCMD>)))
  "Returns md5sum for a message object of type '<SimpleCMD>"
  "b8550e1a4435edb5f28af9d38f17a466")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SimpleCMD)))
  "Returns md5sum for a message object of type 'SimpleCMD"
  "b8550e1a4435edb5f28af9d38f17a466")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SimpleCMD>)))
  "Returns full string definition for message of type '<SimpleCMD>"
  (cl:format cl:nil "  int32 cmd_code~%  int32 cmd_value~%  int32 type~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SimpleCMD)))
  "Returns full string definition for message of type 'SimpleCMD"
  (cl:format cl:nil "  int32 cmd_code~%  int32 cmd_value~%  int32 type~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SimpleCMD>))
  (cl:+ 0
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SimpleCMD>))
  "Converts a ROS message object to a list"
  (cl:list 'SimpleCMD
    (cl:cons ':cmd_code (cmd_code msg))
    (cl:cons ':cmd_value (cmd_value msg))
    (cl:cons ':type (type msg))
))
