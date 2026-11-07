from pydantic import BaseModel, ConfigDict, Field

class Flow(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    PROTOCOL: int = Field(ge=0, le=255, description='IP protocol number')
    TCP_FLAGS: int = Field(ge=0, le=255, description='TCP flags bitmask')
    L7_PROTO: float = Field(ge=0, le=1000000000000)
    IN_BYTES: float = Field(ge=0, le=1000000000000)
    OUT_BYTES: float = Field(ge=0, le=1000000000000)
    IN_PKTS: float = Field(ge=0, le=1000000000000)
    OUT_PKTS: float = Field(ge=0, le=1000000000000)
    FLOW_DURATION_MILLISECONDS: float = Field(ge=0, le=1000000000000)
