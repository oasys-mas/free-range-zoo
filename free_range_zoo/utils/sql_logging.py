"""
SQL logging models and utilities for simulation environments.
Supports PostgreSQL and SQLite backends via SQLAlchemy.
"""
from sqlalchemy import create_engine, Column, Integer, Text, Date, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


class Simulation(Base):
    __tablename__ = 'simulation'

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    description = Column(Text)
    timestamp = Column(Date, nullable=False, default=func.now())

    environments = relationship('Environment', back_populates='simulation')


class Environment(Base):
    __tablename__ = 'environment'

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey('simulation.id'), nullable=False)
    simulation_index = Column(Integer)

    simulation = relationship('Simulation', back_populates='environments')
    agents = relationship('Agent', back_populates='environment')
    timesteps = relationship('EnvironmentTimestep', back_populates='environment')


class Agent(Base):
    __tablename__ = 'agent'

    id = Column(Integer, primary_key=True)
    environment_id = Column(Integer, ForeignKey('environment.id'), nullable=False)
    name = Column(Text, nullable=False)

    environment = relationship('Environment', back_populates='agents')
    logs = relationship('AgentLog', back_populates='agent')


class EnvironmentTimestep(Base):
    __tablename__ = 'environment_timestep'

    environment_id = Column(Integer, ForeignKey('environment.id'), nullable=False)
    id = Column(Integer, primary_key=True)
    timestep = Column(Integer)

    environment = relationship('Environment', back_populates='timesteps')
    agent_logs = relationship('AgentLog', back_populates='timestep')
    wildfire_logs = relationship('WildfireEnvironmentLog', back_populates='timestep')
    rideshare_logs = relationship('RideshareEnvironmentLog', back_populates='timestep')
    cybersecurity_logs = relationship('CybersecurityEnvironmentLog', back_populates='timestep')


class WildfireEnvironmentLog(Base):
    __tablename__ = 'wildfire_environment_log'

    id = Column(Integer, primary_key=True)
    simulation_timestep_id = Column(Integer, ForeignKey('environment_timestep.id'), nullable=False)
    timestep = relationship('EnvironmentTimestep', back_populates='wildfire_logs')

    fires = Column(Text)
    intensity = Column(Text)
    fuel = Column(Text)
    suppressants = Column(Text)
    capacity = Column(Text)
    equipment = Column(Text)
    agents = Column(Text)


class RideshareEnvironmentLog(Base):
    __tablename__ = 'rideshare_environment_log'

    id = Column(Integer, primary_key=True)
    simulation_timestep_id = Column(Integer, ForeignKey('environment_timestep.id'), nullable=False)
    agents = Column(Text)
    passengers = Column(Text)
    timestep = relationship('EnvironmentTimestep', back_populates='rideshare_logs')


class CybersecurityEnvironmentLog(Base):
    __tablename__ = 'cybersecurity_environment_log'

    id = Column(Integer, primary_key=True)
    simulation_timestep_id = Column(Integer, ForeignKey('environment_timestep.id'), nullable=False)
    network_state = Column(Text)
    location = Column(Text)
    presence = Column(Text)
    adj_matrix = Column(Text)
    timestep = relationship('EnvironmentTimestep', back_populates='cybersecurity_logs')


class AgentLog(Base):
    __tablename__ = 'agent_log'

    id = Column(Integer, primary_key=True)
    simulation_timestep_id = Column(Integer, ForeignKey('environment_timestep.id'), nullable=False)
    agent_id = Column(Integer, ForeignKey('agent.id'), nullable=False)
    reward = Column(Integer)
    action_field = Column(Integer)
    task_field = Column(Integer)
    action_map = Column(Text)
    observation_map = Column(Text)
    timestep = relationship('EnvironmentTimestep', back_populates='agent_logs')
    agent = relationship('Agent', back_populates='logs')


def get_engine_and_session(connection_string: str):
    engine = create_engine(connection_string)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session
