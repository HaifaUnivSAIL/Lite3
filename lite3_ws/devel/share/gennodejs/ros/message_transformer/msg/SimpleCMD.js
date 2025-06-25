// Auto-generated. Do not edit!

// (in-package message_transformer.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class SimpleCMD {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.cmd_code = null;
      this.cmd_value = null;
      this.type = null;
    }
    else {
      if (initObj.hasOwnProperty('cmd_code')) {
        this.cmd_code = initObj.cmd_code
      }
      else {
        this.cmd_code = 0;
      }
      if (initObj.hasOwnProperty('cmd_value')) {
        this.cmd_value = initObj.cmd_value
      }
      else {
        this.cmd_value = 0;
      }
      if (initObj.hasOwnProperty('type')) {
        this.type = initObj.type
      }
      else {
        this.type = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SimpleCMD
    // Serialize message field [cmd_code]
    bufferOffset = _serializer.int32(obj.cmd_code, buffer, bufferOffset);
    // Serialize message field [cmd_value]
    bufferOffset = _serializer.int32(obj.cmd_value, buffer, bufferOffset);
    // Serialize message field [type]
    bufferOffset = _serializer.int32(obj.type, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SimpleCMD
    let len;
    let data = new SimpleCMD(null);
    // Deserialize message field [cmd_code]
    data.cmd_code = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [cmd_value]
    data.cmd_value = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [type]
    data.type = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 12;
  }

  static datatype() {
    // Returns string type for a message object
    return 'message_transformer/SimpleCMD';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b8550e1a4435edb5f28af9d38f17a466';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
      int32 cmd_code
      int32 cmd_value
      int32 type
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SimpleCMD(null);
    if (msg.cmd_code !== undefined) {
      resolved.cmd_code = msg.cmd_code;
    }
    else {
      resolved.cmd_code = 0
    }

    if (msg.cmd_value !== undefined) {
      resolved.cmd_value = msg.cmd_value;
    }
    else {
      resolved.cmd_value = 0
    }

    if (msg.type !== undefined) {
      resolved.type = msg.type;
    }
    else {
      resolved.type = 0
    }

    return resolved;
    }
};

module.exports = SimpleCMD;
