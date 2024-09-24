import asyncio
import io
import itertools
import json
import logging
import os
from argparse import ArgumentParser
from functools import wraps
from types import MethodType
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import uvicorn
import yaml
from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from networkx import DiGraph
from networkx.drawing.nx_agraph import graphviz_layout
from pydantic import BaseModel
import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Connection(BaseModel):
    from_: str
    to: List[str]

class Pipeline(BaseModel):
    name: str | None
    description: str | None
    connections: List[Connection]


class Tracer:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def update_endpoint(self, inputs, outputs):
        self.inputs.extend(inputs)
        self.outputs.extend(outputs)

    def set_endpoint(self, app: FastAPI, name: str):
        @app.get(f"/{name}", response_class=HTMLResponse)
        async def _set_endpoint():
            html_content = f"""
            <html>
                <head>
                    <script>
                        function updateData() {{
                            fetch('/{name}/data')
                                .then(response => response.json())
                                .then(data => {{
                                    document.getElementById('inputs').textContent = JSON.stringify(data.inputs, null, 2);
                                    document.getElementById('outputs').textContent = JSON.stringify(data.outputs, null, 2);
                                }});
                        }}

                        // Update every 5 seconds
                        setInterval(updateData, 5000);

                        // Initial update
                        updateData();
                    </script>
                </head>
                <body>
                    <h1>Function: {name}</h1>
                    <h2>Inputs</h2>
                    <pre id="inputs"></pre>
                    <h2>Outputs</h2>
                    <pre id="outputs"></pre>
                </body>
            </html>
            """
            return HTMLResponse(content=html_content)

        @app.get(f"/{name}/data")
        async def _get_data():
            return {"inputs": self.inputs, "outputs": self.outputs}

    @staticmethod
    def inspect_args(func, tracer):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Inspect positional arguments
            inputs = []
            for i, arg in enumerate(args):
                if hasattr(arg, 'iter') and not isinstance(arg, str):
                    # If it's an iterator, use tee to create a copy for inspection
                    arg, arg_copy = itertools.tee(arg)
                    inputs.extend(list(arg_copy))

            outputs = func(*args, **kwargs)
            
            if isinstance(outputs, types.GeneratorType) and not isinstance(outputs, str):
                outputs, outputs_copy = itertools.tee(outputs)
                outputs = list(outputs)

            tracer.update_endpoint(inputs, outputs)
            # Call the original function with the original arguments
            return outputs_copy if 'outputs_copy' in locals() else outputs
        return wrapper


def create_app(pipeline_dump, pipeline_dag) -> FastAPI:
    app = FastAPI()

    class AppState:
        def __init__(self):
            self.dag = None
            self.filename = None
            self.tracers = {}

    app.state.app_state = AppState()

    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting up the application")

        if not hasattr(app.state, 'app_state'):
            app.state.app_state = AppState()

        pipeline = parse_pipeline(pipeline_dump)
        dag = create_dag(pipeline, pipeline_dag)

        app.state.app_state.dag = dag
        app.state.app_state.filename = os.environ['batch_file']

        logger.info(f"DAG is initialized in startup event with {len(app.state.app_state.dag.nodes())} nodes")

        for node in app.state.app_state.dag.nodes():
            if tracer := app.state.app_state.dag.nodes[node].get('tracer'):
                logger.info(f"setting endpoint for node {node}")
                tracer.set_endpoint(app, node)
                app.state.app_state.tracers[node] = tracer

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

        # Create HTML with embedded SVG and JavaScript for clickable areas
        svg_str = buf.getvalue().decode('utf-8')

        for i, (node, (x, y)) in enumerate(pos.items()):
            link_text = f'<a xlink:href="/{node}" target="_blank"><text x="{650}" y="{82+357*i}" fill="transparent" font-size="25px">{node}</text></a>'
            svg_str = svg_str.replace('</svg>', f'{link_text}</svg>')

        # Prepare node data for JavaScript
        # node_data = json.dumps({node: {'x': float(x), 'y': float(y)} for node, (x, y) in pos.items()})

        html_content = f"""
        <html lang="en">
        <head>
            <title>Pipeline DAG Visualization (Debug)</title>
            <style>
                a:hover text {{
                    fill: rgba(255, 0, 0, 0.2) !important;
                }}
            </style>
        </head>
        <body>
            <h1>Pipeline DAG Visualization (Debug)</h1>
            <p>Click on a node to view its traces. Clickable areas are now visible for debugging.</p>
            {svg_str}
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

def create_dag(pipeline: Pipeline, pipeline_dag) -> Dict[str, Any]:
    def wrap_method(obj, method_name, wrapper):
        # original_validate_assignment = obj.__class__.model_config.get('validate_assignment', False)
        # obj.__class__.model_config['validate_assignment'] = False

        # original_method = getattr(obj, method_name)
        wrapped_method = wrapper(obj.__class__.process)
        setattr(obj.__class__, method_name, MethodType(wrapped_method, obj))

        # obj.__class__.model_config['validate_assignment'] = original_validate_assignment

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

        # Wrap node's process function with Tracer
        logger.info(f"{G.nodes[node].values()}")

        if hasattr(pipeline_dag.dag.G.nodes[node]['step'], 'process'):
            logger.info(f"Wrapping process function with Tracer for node {node}")
            tracer = Tracer()
            try:
                step = pipeline_dag.dag.G.nodes[node]['step']
                wrap_method(step, 'process', lambda f: Tracer.inspect_args(f, tracer))
                G.nodes[node]['tracer'] = tracer
                logger.info(f"Wrapped process function for node {node} with tracer {tracer}")
            except Exception as e:
                logger.error(f"Error wrapping process function with Tracer: {e}")
    return G

def load_yaml(file_path: str):
    logger.info(f"Loading YAML from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def parse_pipeline(data: Dict[str, Any]) -> Pipeline:
    logger.info("Parsing pipeline data")

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

def get_app_state(app: FastAPI = Depends()):
    return app.state.app_state

def run_fastapi_app(pipeline_dump, pipeline_dag, batch_file):
    os.environ['batch_file'] = batch_file
    app = create_app(pipeline_dump, pipeline_dag)
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

def run_ui_background(pipeline):
    # import multiprocessing as mp
    import threading

    pipeline_dump = pipeline.dag.dump()
    batch_path = os.path.dirname(pipeline._cache_location['pipeline'])

    # use multiprocessing as threading causes problems with pipeline completion
    # mp.Process(target=run_fastapi_app, args=(pipeline_dump, pipeline, batch_path), daemon=True).start()
    fastapi_thread = threading.Thread(target=run_fastapi_app, args=(pipeline_dump, pipeline, batch_path), daemon=False)
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
