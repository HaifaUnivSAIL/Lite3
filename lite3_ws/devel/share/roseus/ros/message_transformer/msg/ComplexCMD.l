;; Auto-generated. Do not edit!


(when (boundp 'message_transformer::ComplexCMD)
  (if (not (find-package "MESSAGE_TRANSFORMER"))
    (make-package "MESSAGE_TRANSFORMER"))
  (shadow 'ComplexCMD (find-package "MESSAGE_TRANSFORMER")))
(unless (find-package "MESSAGE_TRANSFORMER::COMPLEXCMD")
  (make-package "MESSAGE_TRANSFORMER::COMPLEXCMD"))

(in-package "ROS")
;;//! \htmlinclude ComplexCMD.msg.html


(defclass message_transformer::ComplexCMD
  :super ros::object
  :slots (_cmd_code _cmd_value _type _data ))

(defmethod message_transformer::ComplexCMD
  (:init
   (&key
    ((:cmd_code __cmd_code) 0)
    ((:cmd_value __cmd_value) 0)
    ((:type __type) 0)
    ((:data __data) 0.0)
    )
   (send-super :init)
   (setq _cmd_code (round __cmd_code))
   (setq _cmd_value (round __cmd_value))
   (setq _type (round __type))
   (setq _data (float __data))
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
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; int32 _cmd_code
    4
    ;; int32 _cmd_value
    4
    ;; int32 _type
    4
    ;; float64 _data
    8
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
     ;; float64 _data
       (sys::poke _data (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
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
   ;; float64 _data
     (setq _data (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(setf (get message_transformer::ComplexCMD :md5sum-) "a72ae23e40d8c196548af30d3a5c3007")
(setf (get message_transformer::ComplexCMD :datatype-) "message_transformer/ComplexCMD")
(setf (get message_transformer::ComplexCMD :definition-)
      "  int32 cmd_code
  int32 cmd_value
  int32 type
  float64 data
")



(provide :message_transformer/ComplexCMD "a72ae23e40d8c196548af30d3a5c3007")


