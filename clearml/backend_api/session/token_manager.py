import sys
from abc import ABCMeta, abstractmethod
from time import time
from typing import Optional, Any

import jwt
from jwt.algorithms import get_default_algorithms
import six


@six.add_metaclass(ABCMeta)
class TokenManager(object):
    @property
    def token_expiration_threshold_sec(self) -> int:
        return self.__token_expiration_threshold_sec

    @token_expiration_threshold_sec.setter
    def token_expiration_threshold_sec(self, value: int) -> None:
        self.__token_expiration_threshold_sec = value

    @property
    def req_token_expiration_sec(self) -> int:
        """Token expiration sec requested when refreshing token"""
        return self.__req_token_expiration_sec

    @req_token_expiration_sec.setter
    def req_token_expiration_sec(self, value: Optional[int]) -> None:
        assert isinstance(value, (type(None), int))
        self.__req_token_expiration_sec = value

    @property
    def token_expiration_sec(self) -> int:
        return self.__token_expiration_sec

    @property
    def token(self) -> str:
        return self._get_token()

    @property
    def raw_token(self) -> str:
        return self.__token

    def __init__(
        self,
        token: str = None,
        req_token_expiration_sec: int = None,
        token_history: dict = None,
        token_expiration_threshold_sec: int = 60,
        **kwargs: Any
    ) -> None:
        super(TokenManager, self).__init__()
        assert isinstance(token_history, (type(None), dict))
        self.token_expiration_threshold_sec = token_expiration_threshold_sec
        self.req_token_expiration_sec = req_token_expiration_sec
        self._set_token(token)

    def _calc_token_valid_period_sec(self, token: str, exp: int = None, at_least_sec: int = None) -> int:
        if token:
            try:
                exp = exp or self._get_token_exp(token)
                if at_least_sec:
                    at_least_sec = max(at_least_sec, self.token_expiration_threshold_sec)
                else:
                    at_least_sec = self.token_expiration_threshold_sec
                return max(0, (exp - time() - at_least_sec))
            except Exception:
                pass
        return 0

    @classmethod
    def get_decoded_token(cls, token: str) -> dict:
        """Get token expiration time. If not present, assume forever"""
        if hasattr(jwt, "__version__") and jwt.__version__[0] == "1":
            return jwt.decode(
                token,
                verify=False,
                algorithms=get_default_algorithms(),
            )

        return jwt.decode(
            token,
            options=dict(verify_signature=False),
            algorithms=get_default_algorithms(),
        )

    @classmethod
    def _get_token_exp(cls, token: str) -> int:
        """Get token expiration time. If not present, assume forever"""
        return cls.get_decoded_token(token).get("exp", sys.maxsize)

    def _set_token(self, token: str) -> None:
        if token:
            self.__token = token
            self.__token_expiration_sec = self._get_token_exp(token)
        else:
            self.__token = None
            self.__token_expiration_sec = 0

    def get_token_valid_period_sec(self) -> int:
        return self._calc_token_valid_period_sec(self.__token, self.token_expiration_sec)

    def _get_token(self) -> str:
        if self.get_token_valid_period_sec() <= 0:
            self.refresh_token()
        return self.__token

    @abstractmethod
    def _do_refresh_token(self, old_token: str, exp: int = None) -> str:
        pass

    def refresh_token(self) -> None:
        self._set_token(self._do_refresh_token(self.__token, exp=self.req_token_expiration_sec))
