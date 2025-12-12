"""
Converts SQL logged experiments to csv format padding openness with nulls
"""
import os
import pandas as pd
from typing import Any, Dict, Optional, Union
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine

from free_range_zoo.utils.sql_logging import get_engine_and_session
from free_range_zoo.utils.env import AECEnv


@event.listens_for(Engine, "before_cursor_execute")
def intercept_read_only_writes(conn, cursor, statement, parameters, context, executemany):
    if conn.get_execution_options().get("is_readonly"):
        #find writes
        forbidden_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'TRUNCATE']
        clean_statement = statement.strip().upper()

        #!kill if write attempted...
        if any(clean_statement.startswith(kw) for kw in forbidden_keywords):
            raise RuntimeError(f"Write operation '{clean_statement.split()[0]}' blocked in read-only connection.")


class SQLLogConverter:

    def __init__(self, db_path: str, env: Union[str, AECEnv]):
        self.db_path = db_path
        self.engine, self.Session = get_engine_and_session(db_path)

        if isinstance(env, str):
            if '_' in env:
                self.domain = env.split('_')[0].upper()
            else:
                self.domain = env.upper()
        else:
            self.domain = str(env).split('_')[0].upper()

        assert self.domain in [j.upper() for j in os.listdir(os.path.dirname(__file__) + '/../../free_range_zoo/envs')], \
            f"Environment domain '{self.domain}' not recognized."

    def get_episode(self, environment_id: int, simulation_index: Optional[int] = None) -> pd.DataFrame:
        """
        Query the database for all agent and task features across all timesteps for a given environment_id
        
        Args:
            environment_id (int): The ID of the environment/episode to query
            simulation_index (Optional[int]): The simulation index to filter by. If None, fetches it.
        Returns:
            pd.DataFrame: A DataFrame containing merged agent and environment logs per timestep
        """
        #!this is dangerous
        env_table = f"{self.domain}_ENVIRONMENT_LOG"

        get_time = text("SELECT id,timestep FROM ENVIRONMENT_TIMESTEP WHERE environment_id=:env_id")
        get_simulation = text("SELECT simulation_index FROM ENVIRONMENT WHERE id=:env_id")
        get_agents = text("SELECT * FROM AGENT_LOG WHERE simulation_timestep_id IN :timestep_ids")
        get_env_features = text(f"SELECT * FROM {env_table} WHERE simulation_timestep_id IN :timestep_ids")

        with self.engine.connect() as conn:
            conn = conn.execution_options(is_readonly=True)

            if simulation_index is None:
                simulation_index = pd.read_sql(get_simulation, conn, params={
                    "env_id": environment_id
                }).iloc[0]['simulation_index']

            time = pd.read_sql(get_time, conn, params={"env_id": environment_id})
            agents = pd.read_sql(get_agents, conn, params={"timestep_ids": tuple(time['id'].tolist())})
            env_features = pd.read_sql(get_env_features, conn, params={"timestep_ids": tuple(time['id'].tolist())})

        env_features = env_features.drop(columns=['id'])
        time_env = pd.merge(time, env_features, left_on='id', right_on='simulation_timestep_id')

        #creates per agent columns filling missing with NaN
        agents = agents.drop(columns=['id'])
        agent_pivot = agents.pivot(index='simulation_timestep_id', columns='agent_id')
        agent_pivot.columns = ['_'.join(map(str, col)).strip() for col in agent_pivot.columns.values]
        agent_pivot = agent_pivot.reset_index()
        time_env_agents = pd.merge(time_env, agent_pivot, on='simulation_timestep_id', how='left')
        time_env_agents['simulation_index'] = simulation_index

        return time_env_agents
