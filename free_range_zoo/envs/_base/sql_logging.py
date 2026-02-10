"""SQL Logging Models and Utilities for Simulation Environments.

This module provides SQLAlchemy models and utilities for logging simulation
data to PostgreSQL and SQLite databases.
"""

from sqlalchemy import create_engine, Column, Integer, Text, Date, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


class Simulation(Base):
    """SQLAlchemy model representing a simulation run.

    Attributes:
        id: Primary key identifier for the simulation.
        name: Name of the simulation.
        description: Optional description of the simulation.
        timestamp: Date when the simulation was created.
        environments: Related Environment records.
    """

    __tablename__ = 'simulation'

    id = Column(Integer, primary_key=True)
    name = Column(Text, nullable=False)
    description = Column(Text)
    timestamp = Column(Date, nullable=False, default=func.now())

    environments = relationship('Environment', back_populates='simulation')


class Environment(Base):
    """SQLAlchemy model representing an environment instance within a simulation.

    Attributes:
        id: Primary key identifier for the environment.
        simulation_id: Foreign key to the parent Simulation.
        simulation_index: Index of this environment within the simulation.
        simulation: Related Simulation record.
        agents: Related Agent records.
        timesteps: Related EnvironmentTimestep records.
    """

    __tablename__ = 'environment'

    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey('simulation.id'), nullable=False)
    simulation_index = Column(Integer)

    simulation = relationship('Simulation', back_populates='environments')
    agents = relationship('Agent', back_populates='environment')
    timesteps = relationship('EnvironmentTimestep', back_populates='environment')


class Agent(Base):
    """SQLAlchemy model representing an agent within an environment.

    Attributes:
        id: Primary key identifier for the agent.
        environment_id: Foreign key to the parent Environment.
        name: Name of the agent.
        environment: Related Environment record.
        logs: Related AgentLog records.
    """

    __tablename__ = 'agent'

    id = Column(Integer, primary_key=True)
    environment_id = Column(Integer, ForeignKey('environment.id'), nullable=False)
    name = Column(Text, nullable=False)

    environment = relationship('Environment', back_populates='agents')
    logs = relationship('AgentLog', back_populates='agent')


class EnvironmentTimestep(Base):
    """SQLAlchemy model representing a timestep within an environment.

    Attributes:
        environment_id: Foreign key to the parent Environment.
        id: Primary key identifier for the timestep.
        timestep: The step number within the environment.
        environment: Related Environment record.
        agent_logs: Related AgentLog records.
        wildfire_logs: Related WildfireEnvironmentLog records.
        rideshare_logs: Related RideshareEnvironmentLog records.
        cybersecurity_logs: Related CybersecurityEnvironmentLog records.
    """

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
    """SQLAlchemy model for logging wildfire environment state.

    Attributes:
        id: Primary key identifier.
        simulation_timestep_id: Foreign key to EnvironmentTimestep.
        timestep: Related EnvironmentTimestep record.
        fires: String representation of fire locations.
        intensity: String representation of fire intensities.
        fuel: String representation of fuel levels.
        suppressants: String representation of suppressant levels.
        capacity: String representation of capacity values.
        equipment: String representation of equipment states.
        agents: String representation of agent states.
    """

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
    """SQLAlchemy model for logging rideshare environment state.

    Attributes:
        id: Primary key identifier.
        simulation_timestep_id: Foreign key to EnvironmentTimestep.
        agents: String representation of agent states.
        passengers: String representation of passenger states.
        timestep: Related EnvironmentTimestep record.
    """

    __tablename__ = 'rideshare_environment_log'

    id = Column(Integer, primary_key=True)
    simulation_timestep_id = Column(Integer, ForeignKey('environment_timestep.id'), nullable=False)
    agents = Column(Text)
    passengers = Column(Text)
    timestep = relationship('EnvironmentTimestep', back_populates='rideshare_logs')


class CybersecurityEnvironmentLog(Base):
    """SQLAlchemy model for logging cybersecurity environment state.

    Attributes:
        id: Primary key identifier.
        simulation_timestep_id: Foreign key to EnvironmentTimestep.
        network_state: String representation of network state.
        location: String representation of agent locations.
        presence: String representation of agent presence.
        adj_matrix: String representation of adjacency matrix.
        timestep: Related EnvironmentTimestep record.
    """

    __tablename__ = 'cybersecurity_environment_log'

    id = Column(Integer, primary_key=True)
    simulation_timestep_id = Column(Integer, ForeignKey('environment_timestep.id'), nullable=False)
    network_state = Column(Text)
    location = Column(Text)
    presence = Column(Text)
    adj_matrix = Column(Text)
    timestep = relationship('EnvironmentTimestep', back_populates='cybersecurity_logs')


class AgentLog(Base):
    """SQLAlchemy model for logging agent actions and rewards.

    Attributes:
        id: Primary key identifier.
        simulation_timestep_id: Foreign key to EnvironmentTimestep.
        agent_id: Foreign key to Agent.
        reward: Reward received by the agent.
        action_field: Task field of the action taken.
        task_field: Task field of the action taken.
        action_map: String representation of action mapping.
        observation_map: String representation of observation mapping.
        timestep: Related EnvironmentTimestep record.
        agent: Related Agent record.
    """

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
    """Create a SQLAlchemy engine and session factory.

    Args:
        connection_string (str): SQLAlchemy connection string (e.g., 'sqlite:///path/to/db')

    Returns:
        Tuple[Engine, sessionmaker]: Tuple of (engine, Session) where engine is the database engine
        and Session is the sessionmaker factory
    """
    engine = create_engine(connection_string)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session
