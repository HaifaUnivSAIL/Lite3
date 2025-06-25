
(cl:in-package :asdf)

(defsystem "message_transformer-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "ComplexCMD" :depends-on ("_package_ComplexCMD"))
    (:file "_package_ComplexCMD" :depends-on ("_package"))
    (:file "SimpleCMD" :depends-on ("_package_SimpleCMD"))
    (:file "_package_SimpleCMD" :depends-on ("_package"))
  ))