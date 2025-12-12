"""
Logging handler interface and implementations for simulation environments.
"""
import os
import pandas as pd
from typing import Any, Dict, Optional
from free_range_zoo.utils.sql_logging import (
    get_engine_and_session,
    Simulation,
    Environment as SQLEnvironment,
    Agent as SQLAgent,
    EnvironmentTimestep,
    WildfireEnvironmentLog,
    RideshareEnvironmentLog,
    CybersecurityEnvironmentLog,
    AgentLog,
)
import datetime


class Logger:
    """
    Abstract base logger interface for simulation environments.
    """

    def log_environment(self, *args, **kwargs):
        raise NotImplementedError

    def log_agent(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        pass


class CSVLogger(Logger):
    """
    CSV logger that matches the current output format exactly.
    """

    def __init__(self, log_directory: str, parallel_envs: int, override_initialization_check: bool = False):
        self.log_directory = log_directory
        self.parallel_envs = parallel_envs
        self._are_logs_initialized = False
        self._agent_set = None
        if not override_initialization_check and os.path.exists(log_directory):
            if os.listdir(log_directory):
                raise FileExistsError("The logging output directory already exists. Set override_initialization_check or rename.")
        if not os.path.exists(log_directory):
            os.mkdir(log_directory)

    def log_environment(self,
                        state,
                        actions,
                        rewards,
                        agent_action_mapping,
                        agent_observation_mapping,
                        num_moves,
                        finished,
                        log_description,
                        agents,
                        extra=None,
                        reset=False):
        if extra is not None and len(extra) != self.parallel_envs:
            raise ValueError('The number of elements in extras must match the number of parallel environments.')

        # throw an error if agents have been added mid-simulation and we are using csv loggers
        agent_tuple = tuple(agents)
        if self._agent_set is None or reset:
            self._agent_set = agent_tuple
        elif agent_tuple != self._agent_set:
            raise RuntimeError(f"CSVLogger does not support changing agents mid-simulation. "
                               f"Initial agents: {self._agent_set}, current agents: {agent_tuple}.")

        if reset:
            self._are_logs_initialized = False

        df = state.to_dataframe()
        new_cols = {}
        if reset:
            for agent in agents:
                new_cols[f'{agent}_action'] = [None] * len(df)
                new_cols[f'{agent}_rewards'] = [None] * len(df)
            new_cols['step'] = [-1] * len(df)
            new_cols['complete'] = [None] * len(df)
        else:
            for agent in agents:
                new_cols[f'{agent}_action'] = [str(action) for action in actions[agent].cpu().tolist()]
                new_cols[f'{agent}_rewards'] = rewards[agent].cpu().tolist()
            new_cols['step'] = num_moves.cpu().tolist()
            new_cols['complete'] = finished.cpu().tolist()

        for agent in agents:
            new_cols[f'{agent}_action_map'] = [str(mapping.tolist()) for mapping in agent_action_mapping[agent]]
            new_cols[f'{agent}_observation_map'] = [str(mapping.tolist()) for mapping in agent_observation_mapping[agent]]

        df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
        if extra is not None:
            df = pd.concat([df, extra], axis=1)

        df['description'] = log_description
        for i in range(self.parallel_envs):
            df.iloc[[i]].to_csv(
                os.path.join(self.log_directory, f'{i}.csv'),
                mode='w' if not self._are_logs_initialized else 'a',
                header=not self._are_logs_initialized,
                index=False,
                na_rep="NULL",
            )

        self._are_logs_initialized = True

    def reset(self, *args, **kwargs):
        self._are_logs_initialized = False


class SQLLogger(Logger):
    """
    SQL logger for simulation environments. Uses domain-specific tables.
    """

    def __init__(self, connection_string: str, domain: str, parallel_envs: int):
        self.connection_string = connection_string
        self.domain = domain
        self.parallel_envs = parallel_envs
        self.engine, self.Session = get_engine_and_session(connection_string)
        self.session = self.Session()
        self._simulation_id = None
        self._env_ids = None
        self._agent_ids = None

    def reset(self, log_label=None, log_description=None, agents=None):
        session = self.Session()
        with session.begin():
            sim = Simulation(
                name=log_label or 'simulation',
                description=log_description,
                timestamp=datetime.datetime.now().date(),
            )
            session.add(sim)
            session.flush()

            self._simulation_id = sim.id

            self._env_ids = []
            for i in range(self.parallel_envs):
                env_rec = SQLEnvironment(simulation_id=sim.id, simulation_index=i)
                session.add(env_rec)
                session.flush()
                self._env_ids.append(env_rec.id)

            self._agent_ids = {}
            if agents is not None:
                for agent in agents:
                    for env_id in self._env_ids:
                        agent_rec = SQLAgent(name=agent, environment_id=env_id)
                        session.add(agent_rec)
                        session.flush()
                        self._agent_ids[(agent, env_id)] = agent_rec.id

    def log_environment(self,
                        state,
                        actions,
                        rewards,
                        agent_action_mapping,
                        agent_observation_mapping,
                        num_moves,
                        finished,
                        log_description,
                        agents,
                        extra=None,
                        reset=False):
        if self._env_ids is None:
            raise RuntimeError("SQLLogger: reset() must be called before logging. _env_ids is None.")

        try:
            with self.session.begin():
                for env_idx, env_id in enumerate(self._env_ids):
                    timestep_rec = EnvironmentTimestep(environment_id=env_id, timestep=int(num_moves[env_idx].item()))
                    self.session.add(timestep_rec)
                    self.session.flush()

                    domain = self.domain.split('_')[0]

                    match domain:
                        case 'wildfire':
                            env_log = WildfireEnvironmentLog(
                                simulation_timestep_id=timestep_rec.id,
                                fires=str(getattr(state, 'fires', None)[env_idx].tolist()),
                                intensity=str(getattr(state, 'intensity', None)[env_idx].tolist()),
                                fuel=str(getattr(state, 'fuel', None)[env_idx].tolist()),
                                suppressants=str(getattr(state, 'suppressants', None)[env_idx].tolist()),
                                capacity=str(getattr(state, 'capacity', None)[env_idx].tolist()),
                                equipment=str(getattr(state, 'equipment', None)[env_idx].tolist()),
                                agents=str(getattr(state, 'agents', None)[env_idx].tolist()),
                            )
                        case 'rideshare':
                            env_log = RideshareEnvironmentLog(
                                simulation_timestep_id=timestep_rec.id,
                                agents=str(getattr(state, 'agents', None)[env_idx].tolist()),
                                passengers=str(getattr(state, 'passengers', None)[env_idx].tolist()),
                            )
                        case 'cybersecurity':
                            env_log = CybersecurityEnvironmentLog(
                                simulation_timestep_id=timestep_rec.id,
                                network_state=str(getattr(state, 'network_state', None)[env_idx].tolist()),
                                location=str(getattr(state, 'location', None)[env_idx].tolist()),
                                presence=str(getattr(state, 'presence', None)[env_idx].tolist()),
                            )
                        case _:
                            raise NotImplementedError(f"Environment {self.domain} does not have an implemented log_environment function.")
                    self.session.add(env_log)

                    for agent in agents:
                        # add any new agents which were not in possible_agents before to the simulation
                        if (agent, env_id) not in self._agent_ids:
                            agent_rec = SQLAgent(name=agent, environment_id=env_id)
                            self.session.add(agent_rec)
                            self.session.flush()
                            self._agent_ids[(agent, env_id)] = agent_rec.id

                        agent_id = self._agent_ids.get((agent, env_id))
                        if agent_id is not None:
                            agent_log = AgentLog(
                                simulation_timestep_id=timestep_rec.id,
                                agent_id=agent_id,
                                reward=int(rewards[agent][env_idx].item()),
                                action_field=int(actions[agent][env_idx][0].item()),
                                task_field=int(actions[agent][env_idx][1].item()),
                                action_map=str(agent_action_mapping[agent][env_idx].tolist()),
                                observation_map=str(agent_observation_mapping[agent][env_idx].tolist()),
                            )
                            self.session.add(agent_log)

        except Exception as e:
            self.session.rollback()
            raise e
