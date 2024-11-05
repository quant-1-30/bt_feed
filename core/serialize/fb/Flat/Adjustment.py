# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Flat

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Adjustment(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Adjustment()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsAdjustment(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Adjustment
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Adjustment
    def Sid(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Adjustment
    def RegisterDate(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Adjustment
    def ExDate(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Adjustment
    def Share(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Adjustment
    def Transfer(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Adjustment
    def Interest(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def AdjustmentStart(builder):
    builder.StartObject(6)

def Start(builder):
    AdjustmentStart(builder)

def AdjustmentAddSid(builder, sid):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(sid), 0)

def AddSid(builder, sid):
    AdjustmentAddSid(builder, sid)

def AdjustmentAddRegisterDate(builder, registerDate):
    builder.PrependInt32Slot(1, registerDate, 0)

def AddRegisterDate(builder, registerDate):
    AdjustmentAddRegisterDate(builder, registerDate)

def AdjustmentAddExDate(builder, exDate):
    builder.PrependInt32Slot(2, exDate, 0)

def AddExDate(builder, exDate):
    AdjustmentAddExDate(builder, exDate)

def AdjustmentAddShare(builder, share):
    builder.PrependInt32Slot(3, share, 0)

def AddShare(builder, share):
    AdjustmentAddShare(builder, share)

def AdjustmentAddTransfer(builder, transfer):
    builder.PrependInt32Slot(4, transfer, 0)

def AddTransfer(builder, transfer):
    AdjustmentAddTransfer(builder, transfer)

def AdjustmentAddInterest(builder, interest):
    builder.PrependInt32Slot(5, interest, 0)

def AddInterest(builder, interest):
    AdjustmentAddInterest(builder, interest)

def AdjustmentEnd(builder):
    return builder.EndObject()

def End(builder):
    return AdjustmentEnd(builder)
