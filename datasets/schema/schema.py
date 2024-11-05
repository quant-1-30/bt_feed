#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import uuid
from typing import List
from typing import Optional
from sqlalchemy import func
from sqlalchemy import Integer, String, ForeignKey, BigInteger, Text, UUID
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.schema import PrimaryKeyConstraint


# declarative base class
class Base(DeclarativeBase):
    pass


class User(Base):

    __tablename__ = "user_account"
    __table_args__ = {"extend_existing": True}
    # __table_args__ = (PrimaryKeyConstraint("id", "***"),)

    # 唯一主键
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # primary = unique + not null
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(30), use_existing_column=True)
    # 计算默认值用server_default --- dababase 非default --- python
    # 默认日期 func.now / func.current_timestamp()
    register_time: Mapped[datetime.datetime] = mapped_column(server_default=func.now(), use_existing_column=True)
    fullname: Mapped[Optional[str]] = mapped_column(default="")
    phone: Mapped[Optional[int]] = mapped_column(BigInteger, unique=True)

    # backref在主类里面申明 / back_populates显式两个类申明
    addresses: Mapped[List["Address"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    experiment: Mapped[List["Experiment"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"User(id={self.id!r}), name={self.name!r}, fullname={self.fullname!r}"


class Experiment(Base):

    # 增加account_id 与 user_id , algo_id映射关系表
    __tablename__ = "experiment"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # foreignKey --- unique
    user_id: Mapped[str] = mapped_column(ForeignKey("user_account.user_id", ondelete="CASCADE", onupdate="CASCADE"), use_existing_column=True)
    account_id: Mapped[int] = mapped_column(String(64), nullable=False)
    experiment_id: Mapped[int] = mapped_column(String(20), nullable=False, use_existing_column=True)

    user: Mapped["User"] = relationship(back_populates="experiment")
    order: Mapped[List["Order"]] = relationship(back_populates="experiment")
    transaction: Mapped[List["Transaction"]] = relationship(back_populates="experiment")
    account: Mapped[List["Account"]] = relationship(back_populates="experiment")


class Address(Base):

    __tablename__ = "address"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email_address: Mapped[str]
    user_id: Mapped[int] = mapped_column(ForeignKey("user_account.id", ondelete="CASCADE"), use_existing_column=True)

    user: Mapped["User"] = relationship(back_populates="addresses")

    def __repr__(self) -> str:
        return f"Address(id={self.id!r}, email_address={self.email_address!r})"
    

class Calendar(Base):

    __tablename__ = "calendar"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trading_date: Mapped[int] = mapped_column(Integer,unique=True, nullable=False, primary_key=True, use_existing_column=True)


class Instrument(Base):


    __tablename__ = "instrument"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sid: Mapped[str] = mapped_column(String(10), unique=True, nullable=False, primary_key=True)
    name: Mapped[str] = mapped_column(String(25), nullable=False, primary_key=True)
    first_trading: Mapped[int] = mapped_column(Integer, nullable=False)
    delist: Mapped[int] = mapped_column(Integer, default=0)


class Line(Base):
   
    __tablename__ = "minute"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sid: Mapped[str] = mapped_column(String(20), 
                                     ForeignKey("instrument.sid", onupdate="CASCADE", ondelete="CASCADE"), 
                                     nullable=False, primary_key=True, use_existing_column=True)
    tick: Mapped[int] = mapped_column(BigInteger, nullable=False, primary_key=True, use_existing_column=True)
    open: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    high: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    low: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    close: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, use_existing_column=True)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False, use_existing_column=True)


class Adjustment(Base):

    # register_date:登记日 ; ex_date:除权除息日 ; pay_date:除权除息日 ; effective_date:上市日期
    # 股权登记日后的下一个交易日就是除权日或除息日，这一天购入该公司股票的股东不再享有公司此次分红配股
    # 上交所证券的红股上市日为股权除权日的下一个交易日; 深交所证券的红股上市日为股权登记日后的第3个交易日
    # share --- 送股 / transfer --- 转股 / interest --- 股息

    __tablename__ = "adjustment"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sid: Mapped[str] = mapped_column(String(20), 
                                     ForeignKey("instrument.sid", onupdate="CASCADE", ondelete="CASCADE"), 
                                     nullable=False, primary_key=True, use_existing_column=True)
    register_date: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    ex_date: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    bonus_share: Mapped[int] = mapped_column(Integer, nullable=True, use_existing_column=True)
    transfer_share: Mapped[int] = mapped_column(Integer, nullable=True, use_existing_column=True)
    interest: Mapped[int] = mapped_column(Integer, nullable=True, use_existing_column=True)


class Rightment(Base):

    # register_date:登记日 ; ex_date:除权除息日 
    # 股权登记日后的下一个交易日就是除权日或除息日，这一天购入该公司股票的股东不再享有公司此次分红配股
    # 上交所证券的红股上市日为股权除权日的下一个交易日; 深交所证券的红股上市日为股权登记日后的第3个交易日
    # price --- 配股价格 / ratio --- 配股比例

    __tablename__ = "rightment"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sid: Mapped[str] = mapped_column(String(20), 
                                     ForeignKey("instrument.sid", onupdate="CASCADE", ondelete="CASCADE"), 
                                     nullable=False, primary_key=True, use_existing_column=True)
    register_date: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    ex_date: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    effective_date: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    price: Mapped[int] = mapped_column(Integer, nullable=True, use_existing_column=True)
    ratio: Mapped[int] = mapped_column(Integer, nullable=True, use_existing_column=True)


class Order(Base):

    __tablename__ = "order"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sid: Mapped[str] = mapped_column(String(10), nullable=False, use_existing_column=True)
    created_dt: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    order_id: Mapped[int] = mapped_column(String(16), primary_key=True, nullable=False, unique=True, use_existing_column=True, )
    order_type: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    price: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    volume: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)

    experiment: Mapped["Experiment"] = relationship(
        back_populates="order", cascade="all, delete-orphan")

    transaction: Mapped[List["Transaction"]] = relationship(
        # uselist False -对一
        back_populates="order", cascade="all, delete-orphan"
    )


class Transaction(Base):

    __tablename__ = "transaction"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sid: Mapped[str] = mapped_column(String(10), primary_key=True, nullable=False, use_existing_column=True)
    created_at: Mapped[int] = mapped_column(Integer, primary_key=True,nullable=False, use_existing_column=True)
    price: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, use_existing_column=True)
    cost: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)

    experiment: Mapped["Experiment"] = relationship(
        back_populates="transaction", cascade="all, delete-orphan")
    order: Mapped["Order"] = relationship(
        back_populates="transaction", cascade="all, delete-orphan")


class Account(Base):

    __tablename__ = "account"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    date: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True, nullable=False)
    positions: Mapped[str] = mapped_column(Text, nullable=False, use_existing_column=True)
    portfolio: Mapped[int] = mapped_column(BigInteger, nullable=False, use_existing_column=True)
    cash: Mapped[int] = mapped_column(Integer, nullable=False, use_existing_column=True)

    experiment: Mapped["Experiment"] = relationship(
        back_populates="account", cascade="all, delete-orphan")
