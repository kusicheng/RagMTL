                                                                           
from dataclasses import dataclass, field
from pathlib import Path
import re


class ContinuityError(RuntimeError):
    pass                                                                          


def database_name(value):
    if not isinstance(value, str) or len(value) > 64 or not re.fullmatch(r"magra_continuity(?:_[a-z0-9_]+)?", value):
        raise ContinuityError("dedicated_database_name_required")
    return value


@dataclass(frozen=True)
class MySQLConfig:
    password: str = field(repr=False)
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    database: str = "magra_continuity"

    def __post_init__(self):
        if not isinstance(self.password, str) or not self.password or len(self.password.encode("utf-8")) > 16384 or any(c in self.password for c in "\r\n\0"):
            raise ContinuityError("mysql_password_required")
        if not isinstance(self.host, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,253}", self.host):
            raise ContinuityError("invalid_mysql_host")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ContinuityError("invalid_mysql_port")
        if not isinstance(self.user, str) or not self.user or len(self.user) > 128 or any(ord(c) < 32 for c in self.user):
            raise ContinuityError("invalid_mysql_user")
        database_name(self.database)


def parse_mysql_config(text):
    pass                                                            

                                                                             
                                                                                
       
    if not isinstance(text, str):
        raise ContinuityError("invalid_config_text")
    allowed = {"mysql", "MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_DATABASE"}
    values = {}
    for line in text.lstrip("\ufeff").splitlines():
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in allowed:
            continue
        if name in values:
            raise ContinuityError("duplicate_mysql_config_field")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    try:
        port = int(values.get("MYSQL_PORT", "3306"))
    except ValueError:
        raise ContinuityError("invalid_mysql_port") from None
    return MySQLConfig(password=values.get("mysql", ""), host=values.get("MYSQL_HOST", "127.0.0.1"),
                       port=port, user=values.get("MYSQL_USER", "root"),
                       database=values.get("MYSQL_DATABASE", "magra_continuity"))


def load_mysql_config(path):
    try:
        raw = Path(path).read_bytes()
        if len(raw) > 1_000_000:
            raise ContinuityError("config_file_too_large")
        return parse_mysql_config(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError):
        raise ContinuityError("config_file_unavailable") from None
