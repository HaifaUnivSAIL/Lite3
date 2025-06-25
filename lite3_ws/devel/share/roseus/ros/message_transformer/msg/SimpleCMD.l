;; Auto-generated. Do not edit!


(when (boundp 'message_transformer::SimpleCMD)
  (if (not (find-package "MESSAGE_TRANSFORMER"))
    (make-package "MESSAGE_TRANSFORMER"))
  (shadow 'SimpleCMD (find-package "MESSAGE_TRANSFORMER")))
(unless (find-package "MESSAGE_TRANSFORMER::SIMPLECMD")
  (make-package "MESSAGE_TRANSFORMER::SIMPLECMD"))

(in-package "ROS")
;;//! \htmlinclude SimpleCMD.msg.html


(defclass message_transformer::SimpleCMD
  :super ros::object
  :slots (_cmd_code _cmd_value _type ))

(defmethod message_transformer::SimpleCMD
  (:init
   (&key
    ((:cmd_code __cmd_code) 0)
    ((:cmd_value __cmd_value) 0)
    ((:type __type) 0)
    )
   (send-super :init)
   (setq _cmd_code (round __cmd_code))
   (setq _cmd_value (round __cmd_value))
   (setq _type (round __type))
   self)
  (:cmd_code
   (&optional __cmd_code)
   (if __cmd_code (setq _cmd_code __cmd_code)) _cmd_code)
  (:cmd_value
   (&optional __cmd_value)
   (if __cmd_value (setq _cmd_value __cmd_value)) _cmd_value)
  (:type
   (&optional __type)
   (if __type (setq _type __type)) _type)
  (:serialization-length
   ()
   (+
    ;; int32 _cmd_code
    4
    ;; int32 _cmd_value
    4
    ;; int32 _type
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int32 _cmd_code
       (write-long _cmd_code s)
     ;; int32 _cmd_value
       (write-long _cmd_value s)
     ;; int32 _type
       (write-long _type s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int32 _cmd_code
     (setq _cmd_code (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _cmd_value
     (setq _cmd_value (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int32 _type
     (setq _type (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get message_transformer::SimpleCMD :md5sum-) "b8550e1a4435edb5f28af9d38f17a466")
(setf (get message_transformer::SimpleCMD :datatype-) "message_transformer/SimpleCMD")
(setf (get message_transformer::SimpleCMD :definition-)
      "  int32 cmd_code
  int32 cmd_value
  int32 type
")



(provide :message_transformer/SimpleCMD "b8550e1a4435edb5f28af9d38f17a466")


