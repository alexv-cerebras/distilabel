import asyncio
import io
import json
import os
import logging
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import uvicorn
import yaml
from fastapi import FastAPI, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from networkx import DiGraph
from networkx.drawing.nx_agraph import graphviz_layout
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Connection(BaseModel):
    from_: str
    to: List[str]

class Pipeline(BaseModel):
    name: str | None
    description: str | None
    connections: List[Connection]
        
        
def create_app(pipeline_dump) -> FastAPI:
    app = FastAPI()
    
    class AppState:
        def __init__(self):
            self.dag = None
            self.filename = None

    app.state.app_state = AppState()

    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting up the application")

        if not hasattr(app.state, 'app_state'):
            app.state.app_state = AppState()

        pipeline = parse_pipeline(pipeline_dump)
        dag = create_dag(pipeline)
        app.state.app_state.dag = dag
        app.state.app_state.filename = os.environ['batch_file']

        logger.info(f"DAG is initialized in startup event with {len(app.state.app_state.dag.nodes())} nodes")

        app.state.update_task = asyncio.create_task(update_dag(app))
        
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down the application")
        if hasattr(app.state, 'update_task'):
            app.state.update_task.cancel()
            try:
                await app.state.update_task
            except asyncio.CancelledError:
                pass
            
    @app.get("/debug")
    async def debug_info(request: Request):
        state = request.app.state.app_state if hasattr(request.app.state, 'app_state') else None
        return JSONResponse({
            "app_state_exists": state is not None,
            "dag_initialized": state.dag is not None if state else False,
            "filename": state.filename if state else None,
            "nodes": list(state.dag.nodes()) if state and state.dag else None,
            "edges": list(state.dag.edges()) if state and state.dag else None,
        })
        
    @app.get("/dag/svg", response_class=HTMLResponse)
    async def get_dag_svg(request: Request):
        def is_leaf_node(G, node):
            return G.out_degree(node) == 0
        
        state = request.app.state.app_state
        if not state.dag:
            return HTMLResponse(content="<p>DAG not initialized</p>")
        
        plt.figure(figsize=(20, 12))
        pos = graphviz_layout(state.dag, prog='dot', args='-Grankdir=TB -Gnodesep=0.5 -Granksep=1.0')

        # Node colors based on type
        node_colors = ['#87CEFA' if is_leaf_node(state.dag, node) else '#98FB98' for node in state.dag.nodes()]

        nx.draw(state.dag, pos, with_labels=False, node_color=node_colors, node_size=7000,
                arrows=True, edge_color='gray', width=1, arrowsize=20)

        # Add labels with pipeline step in bold and seq_no underneath
        for node, (x, y) in pos.items():
            plt.text(x, y+10, node, ha='center', va='center', fontweight='bold', fontsize=10, wrap=True)
            plt.text(x, y-20, f"number of processed batches: {state.dag.nodes[node]['seq_no']}", ha='center', va='center', fontsize=8)

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close()

        # Create HTML with embedded SVG
        svg_str = buf.getvalue().decode('utf-8')
        html_content = f"""
        <html>
            <body>
                <h1>Pipeline DAG Visualization</h1>
                {svg_str}
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    @app.get("/dag/visualize", response_class=HTMLResponse)
    async def visualize_dag():
        html_content = """
        <html>
            <head>
                <script>
                    function updateSVG() {
                        fetch('/dag/svg')
                            .then(response => response.text())
                            .then(svg => {
                                document.getElementById('dag-container').innerHTML = svg;
                            });
                    }

                    // Update every 5 seconds
                    setInterval(updateSVG, 5000);

                    // Initial update
                    updateSVG();
                </script>
            </head>
            <body>
                <div id="dag-container"></div>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)
            
    async def update_dag(app):
        while True:
            if app.state.app_state.dag:
                logger.info("Updating DAG")
                for node in app.state.app_state.dag.nodes():
                    data = load_batch_manager_data(os.environ['batch_file'], node)
                    if data:
                        app.state.app_state.dag.nodes[node]['seq_no'] = data['seq_no']
            else:
                logger.warning("DAG not initialized in update_dag")
            await asyncio.sleep(5)  # Update every 5 seconds
    return app

def load_yaml(file_path: str):
    logger.info(f"Loading YAML from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def parse_pipeline(data: Dict[str, Any]) -> Pipeline:
    logger.info(f"Parsing pipeline data")
    
    connections = []
    for conn in data.get('connections', []):
        connections.append(Connection(from_=conn.get('from', ''), to=conn.get('to', [])))

    return Pipeline(
        name='',
        description='',
        connections=connections
    )

def load_batch_manager_data(filename, step_name):
    file_path = f"{filename}/batch_manager_steps/{step_name}/batch_manager_step.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    return None

def create_dag(pipeline: Pipeline) -> Dict[str, Any]:
    logger.info("Creating DAG")
    G = DiGraph()
    
    for conn in pipeline.connections:
        for to_node in conn.to:
            G.add_edge(conn.from_, to_node)

    for node in G.nodes():
        data = load_batch_manager_data(os.environ['batch_file'], node)
        if data:
            G.nodes[node]['seq_no'] = data['seq_no']
        else:
            G.nodes[node]['seq_no'] = 0
    return G

def get_app_state(app: FastAPI = Depends()):
    return app.state.app_state

def run_fastapi_app(pipeline_dump, batch_file):
    os.environ['batch_file'] = batch_file
    app = create_app(pipeline_dump)
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
    
def run_ui_background(pipeline):
    import threading

    pipeline_dump = pipeline.dag.dump()
    batch_path = os.path.dirname(pipeline._cache_location['pipeline'])

    fastapi_thread = threading.Thread(target=run_fastapi_app, args=(pipeline_dump, batch_path), daemon=True)
    fastapi_thread.start()
    return fastapi_thread

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dir_name', type=str)
    args = parser.parse_args()
    
    # Load the YAML data
    pipeline_filename = os.path.join(args.dir_name, 'pipeline.yaml')
    pipeline_dump = load_yaml(pipeline_filename)
    run_fastapi_app(pipeline_dump, args.dir_name)
