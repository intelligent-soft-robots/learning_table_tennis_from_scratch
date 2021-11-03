from collections import OrderedDict
import pathlib
import toml


class BallLauncherConfig:

    """
    Configuration of the ball launcher client.
    All argument values as values between 0 and 1.

    Args:
        ip, port : of the server (installed on the ball launcher Rasperri Pi)
        phi : azimutal angle
        theta : altitute of angle
        top_angular_velocity:  angular velocity of upper wheel
        bottom_angular_velocity: angular velocity of lower wheel
    """

    # warning: order matters
    __slots__ = ("ip","port","phi","theta","top_angular_velocity","bottom_angular_velocity")
    
    def __init__(self,
                 ip: str,
                 port: int,
                 phi: float,
                 theta: float,
                 top_angular_velocity: float,
                 bottom_angular_velocity: float):

        values = locals()
        for s in self.__slots__:
            setattr(self, s, values[s])

    @classmethod
    def from_toml(cls,file_path: pathlib.Path):
        """
        Creates an instance of BallLauncherConfig from a toml
        configuration file which must have the keys
        ip, port, phi, theta, top_angular_velocity, bottom_angular_velocity
        """
        
        if not file_path.is_file():
            raise FileNotFoundError(
                "failed to find the ball launcher "
                "configuration file ".format(file_path)
            )
        data = toml.load(file_path)
        for slot in cls.__slots__:
            if slot not in data.keys():
                raise ValueError("failed to parse toml configuration file "
                                 "for key {}".format(slot))
        def _read(data,key,type_):
            try:
                return type_(data[key])
            except:
                raise TypeError("error reading key {} from {},"
                                " {} expected").format(key,file_path,key)
        args = OrderedDict()
        types = ("str","int","float","float","float","float")
        for index,slot in enumerate(cls.__slots__):
            args.move_to_end(_read(data,slot,types[index]))
        return cls(*list(args.keys()))
        
        
