"""
Converts SQL logged experiments to csv format padding openness with nulls
"""
import os
import pandas as pd
import numpy as np
import warnings
from typing import Any, Dict, Optional, Union
from sqlalchemy import create_engine, text, event
from sqlalchemy import select, MetaData, Table
from sqlalchemy.engine import Engine
from collections import defaultdict

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

    def __init__(self, db_path: str, env: Union[str, 'AECEnv'], agent_name: Optional[str] = None):
        """
        Args:
            db_path (str): Path to the SQL database file
            env (Union[str, AECEnv]): The environment name (e.g., 'wildfire_v0') or environment instance
            agent_name (Optional[str], optional): The name of the agent type in the environment.
                If None, it will be inferred from the environment metadata. Defaults to None.
        """
        self.db_path = db_path
        self.engine, self.Session = get_engine_and_session(db_path)

        if isinstance(env, str):
            if '_' in env:
                self.domain = env.split('_')[0].lower()
            else:
                self.domain = env.lower()
        else:
            self.domain = str(env).split('_')[0].lower()

        assert self.domain in [j.lower() for j in os.listdir(os.path.dirname(__file__) + '/../../free_range_zoo/envs')], \
            f"Environment domain '{self.domain}' not recognized."
        
        if agent_name is None:
            if not hasattr(env, 'metadata'):
                self.agent_name = None
                raise ValueError("If 'env' is not an environment instance, 'agent_name' must be provided.")
            else:
                self.agent_name = env.metadata.get('agent_name', None)
        else:
            self.agent_name = agent_name

        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)

    def __call__(self, output_directory: str, name: Optional[str] = None, 
        desc: Optional[str] = None, simulation_index: Optional[int] = None, verbose: bool = False, *args, **kwargs):
        """
        Args:
            name (Optional[str], optional): Partial Name of the episode to retrieve. Defaults to None.
            desc (Optional[str], optional): Partial Description of the episode to retrieve. Defaults to None.
            simulation_index (Optional[int], optional): Simulation index of the episode to retrieve. Defaults to None.
        
        Writes:
            simulationindex_name \forall matching simulations
                - environment_id.csv \forall matching episodes
        """
        assert any([name is not None, desc is not None, simulation_index is not None]), \
            "At least one of 'name', 'desc', or 'simulation_index' must be provided."

        t_sim = self.metadata.tables['simulation']
        t_env = self.metadata.tables['environment']

        with self.engine.connect() as conn:
            conn = conn.execution_options(is_readonly=True)

            if name:
                stmt = select(t_sim.c.id, t_sim.c.name).where(t_sim.c.name.ilike(f"%{name}%"))
                sim_indices_name = conn.execute(stmt).fetchall()
            
            if desc:
                stmt = select(t_sim.c.id,t_sim.c.name).where(t_sim.c.description.ilike(f"%{desc}%"))
                sim_indices_desc = conn.execute(stmt).fetchall()

            if simulation_index is not None:
                if (name or desc) and verbose:
                    warnings.warn("Both 'simulation_index' and 'name'/'desc' provided. Ignoring 'name'/'desc'.")
                stmt = select(t_sim.c.id,t_sim.c.name).where(t_sim.c.id == simulation_index)
                sim_indices = conn.execute(stmt).fetchall()

            else:
                if name and desc:
                    sim_indices = list(set(sim_indices_name) & set(sim_indices_desc))
                elif name:
                    sim_indices = sim_indices_name
                elif desc:
                    sim_indices = sim_indices_desc
            
            if verbose:
                print("Fetching: ", sim_indices)

            sim_paths = {}

            for sim_ind, sim_name in sim_indices:
                try:
                    os.makedirs(os.path.join(output_directory, f"{sim_ind}_{sim_name}"), exist_ok=False)
                    sim_paths[sim_ind] = os.path.join(output_directory, f"{sim_ind}_{sim_name}")
                except:
                    warnings.warn(f"Output directory for simulation '{sim_ind}_{sim_name}' already exists.")
                    if override_initialization_check:
                        warnings.warn("Override flag set. Continuing and potentially overwriting existing files.")
                    else:
                        raise RuntimeError("To override, set 'override_initialization_check' to True.")

            stmt = select(t_env.c.id, t_env.c.simulation_id, t_env.c.simulation_index).where(
                t_env.c.simulation_id.in_([sim[0] for sim in sim_indices])
            )
            env_episodes = conn.execute(stmt).fetchall()

            for env_id, sim_id, sim_index in env_episodes:
                df = self.get_episode(env_id, simulation_index=sim_index, *args, **kwargs)
                output_path = os.path.join(
                    sim_paths[sim_id], f'{sim_index}.csv'
                )
                df.to_csv(output_path, index=False)
            
                



    

    def get_episode(self, environment_id: int, simulation_index: Optional[int] = None, reindex: bool = False) -> pd.DataFrame:
        """
        Query the database for all agent and task features across all timesteps for a given environment_id

        Args:
            environment_id (int): The environment_id of the episode to retrieve
            simulation_index (Optional[int], optional): The simulation_index of the episode to retrieve.
                If None, it will be queried from the database. Defaults to None.
            reindex (bool, optional): If True, agent IDs will be reindexed [0, ...]
        Returns:
            pd.DataFrame: A DataFrame containing merged agent and environment logs per timesteps
        """
        
        # 1. Access Table Objects (Fail fast if core tables are missing)
        try:
            t_env = self.metadata.tables['environment']
            t_timestep = self.metadata.tables['environment_timestep']
            t_agent_log = self.metadata.tables['agent_log']
        except KeyError as e:
            raise RuntimeError(f"Critical table missing from database: {e}")

        # 2. Safely resolve the dynamic Environment Log table
        env_table_name = f"{self.domain}_environment_log"
        if env_table_name not in self.metadata.tables:
            raise ValueError(f"Table '{env_table_name}' does not exist in the database.")
        
        t_env_log = self.metadata.tables[env_table_name]


        with self.engine.connect() as conn:
            conn = conn.execution_options(is_readonly=True)

            # Query 1: Get Simulation Index (if not provided)
            if simulation_index is None:
                stmt_sim = select(t_env.c.simulation_index).where(t_env.c.id == environment_id)
                simulation_index = conn.execute(stmt_sim).scalar()

            # Query 2: Get Time
            stmt_time = select(t_timestep.c.id, t_timestep.c.timestep).where(t_timestep.c.environment_id == environment_id)
            time = pd.read_sql(stmt_time, conn)

            # but what if no timesteps? (empty query)
            if time.empty:
                return pd.DataFrame()

            timestep_ids = time['id'].tolist()

            # Query 3: Get Agents
            stmt_agents = select(t_agent_log).where(t_agent_log.c.simulation_timestep_id.in_(timestep_ids))
            agents = pd.read_sql(stmt_agents, conn)

            # Query 4: Get Env Features (Dynamic Table)
            stmt_features = select(t_env_log).where(t_env_log.c.simulation_timestep_id.in_(timestep_ids))
            env_features = pd.read_sql(stmt_features, conn)

        
        env_features = env_features.drop(columns=['id'], errors='ignore')
        time_env = pd.merge(time, env_features, left_on='id', right_on='simulation_timestep_id')

        #to help with arbirary agent column renaming
        agents = agents.drop(columns=['id'], errors='ignore').add_prefix('agent_')
        agents = agents.rename(columns={'agent_simulation_timestep_id': 'simulation_timestep_id',
                                      'agent_agent_id': 'agent_id'})
        
        # Create per agent columns filling missing with NaN
        if reindex:
            unique_agents = agents['agent_id'].astype(int).unique()
            map_dict = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_agents))}
            agents['agent_id'] = agents['agent_id'].apply(lambda x: map_dict[x])

        agent_pivot = agents.pivot(index='simulation_timestep_id', columns='agent_id')
        agent_pivot.columns = ['_'.join(map(str, col)).strip() for col in agent_pivot.columns.values]
        agent_pivot = agent_pivot.reset_index()
        
        time_env_agents = pd.merge(time_env, agent_pivot, on='simulation_timestep_id', how='left')
        time_env_agents['simulation_index'] = simulation_index

        remove_agent_holder = {col:''.join(col.split('agent_')) for col in time_env_agents.columns if col.startswith('agent_')}
        time_env_agents.rename(columns=remove_agent_holder, inplace=True)

        ag_naming_dict = {
            col: self.agent_name+'_'+col.split('_')[-1]+'_'+'_'.join(col.split('_')[:-1]) 
            for col in time_env_agents.columns if col in remove_agent_holder.values()
        }

        time_env_agents = time_env_agents.rename(columns = {
            'timestep': 'step'} | ag_naming_dict)
        

        complete_map = defaultdict(lambda : False)
        complete_map[0] = float('nan')
        complete_map[time_env_agents['step'].max().item()] = True
        time_env_agents['complete'] = time_env_agents['step'].map(complete_map)


        return time_env_agents
