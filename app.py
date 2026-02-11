from flask import Flask, render_template, jsonify, redirect, request
from urllib.parse import unquote
from utils.db import TracePreprocessor, DEFAULT_PRICING
from utils.viz import create_scatter_plot, create_task_success_heatmap, create_leaderboard, create_bar_chart, create_completion_tokens_bar_chart, create_missing_runs_heatmap, create_model_timeline_chart
import plotly.utils
import json
import pandas as pd
from datetime import datetime

# Authors from the paper (in order)
AUTHORS = [
    {"name": "Sayash Kapoor", "affiliation": "Princeton University"},
    {"name": "Benedikt Stroebl", "affiliation": "Princeton University"},
    {"name": "Peter Kirgis", "affiliation": "Princeton University"},
    {"name": "Nitya Nadgir", "affiliation": "Independent Researcher"},
    {"name": "Zachary S Siegel", "affiliation": "Princeton University"},
    {"name": "Boyi Wei", "affiliation": "Princeton University"},
    {"name": "Tianci Xue", "affiliation": "The Ohio State University"},
    {"name": "Ziru Chen", "affiliation": "The Ohio State University"},
    {"name": "Felix Chen", "affiliation": "Princeton University"},
    {"name": "Saiteja Utpala", "affiliation": "Microsoft Research"},
    {"name": "Franck Ndzomga", "affiliation": "Independent Researcher"},
    {"name": "Dheeraj Oruganty", "affiliation": "Amazon"},
    {"name": "Sophie Luskin", "affiliation": "Princeton University"},
    {"name": "Kangheng Liu", "affiliation": "Georgetown University"},
    {"name": "Botao Yu", "affiliation": "The Ohio State University"},
    {"name": "Amit Arora", "affiliation": "Georgetown University"},
    {"name": "Dongyoon Hahm", "affiliation": "KAIST"},
    {"name": "Harsh Trivedi", "affiliation": "Stony Brook University"},
    {"name": "Huan Sun", "affiliation": "The Ohio State University"},
    {"name": "Juyong Lee", "affiliation": "KAIST"},
    {"name": "Tengjun Jin", "affiliation": "University of Illinois Urbana-Champaign"},
    {"name": "Yifan Mai", "affiliation": "Stanford University"},
    {"name": "Yifei Zhou", "affiliation": "xAI"},
    {"name": "Yuxuan Zhu", "affiliation": "University of Illinois Urbana-Champaign"},
    {"name": "Rishi Bommasani", "affiliation": "Stanford University"},
    {"name": "Daniel Kang", "affiliation": "University of Illinois Urbana-Champaign"},
    {"name": "Dawn Song", "affiliation": "University of California, Berkeley"},
    {"name": "Peter Henderson", "affiliation": "Princeton University"},
    {"name": "Yu Su", "affiliation": "The Ohio State University"},
    {"name": "Percy Liang", "affiliation": "Stanford University"},
    {"name": "Arvind Narayanan", "affiliation": "Princeton University"},
    {"name": "Stephan Rabanser", "affiliation": "Princeton University"},
    {"name": "Andrew Schwartz", "affiliation": "Cornflower Labs"}
]

# Contributors (everyone else who helped with the project)
CONTRIBUTORS = [
    {"name": "Aymeric Roucher", "affiliation": "Hugging Face"},
    {"name": "Ayush Thakur", "affiliation": "Weights & Biases"},
    {"name": "Hailey Schoelkopf", "affiliation": "Anthropic"},
    {"name": "Iason Gabriel", "affiliation": "Google DeepMind"},
    {"name": "Jelena Luketina", "affiliation": "UK AISI"},
    {"name": "JJ Allaire", "affiliation": "UK AISI"},
    {"name": "Laura Weidinger", "affiliation": "Google DeepMind"},
    {"name": "Madhur Prashant", "affiliation": "Amazon"},
    {"name": "Marius Hobbhahn", "affiliation": "Apollo Research"},
    {"name": "Maximillian Kaufmann", "affiliation": "UK AISI"},
    {"name": "Morgan McGuire", "affiliation": "Weights & Biases"},
    {"name": "Omar Khattab", "affiliation": "MIT"},
    {"name": "Parth Asawa", "affiliation": "UC Berkeley"},
    {"name": "Shreya Shankar", "affiliation": "UC Berkeley"},
    {"name": "Shayne Longpre", "affiliation": "MIT"},
    {"name": "Veniamin Veselovsky", "affiliation": "Princeton University"},
    {"name": "William Isaac", "affiliation": "Google DeepMind"},
    {"name": "Charles Teague", "affiliation": "UK AISI"},
    {"name": "Clémentine Fourrier", "affiliation": "Hugging Face"},
    {"name": "Kevin Meng", "affiliation": "Transluce"}
]


# Agent scaffold links mapping
AGENT_LINKS = {
    "Browser-Use": {
        "type": "GitHub",
        "url": "https://github.com/browser-use/browser-use"
    },
    "CORE-Agent": {
        "type": "GitHub", 
        "url": "https://github.com/siegelz/core-bench/tree/main/agents/AutoGPT-CORE"
    },
    "HAL Generalist Agent": {
        "type": "GitHub",
        "url": "https://github.com/princeton-pli/hal-harness/tree/main/agents/hal_generalist_agent"
    },
    "HF Open Deep Research": {
        "type": "Blog",
        "url": "https://huggingface.co/blog/open-deep-research"
    },
    "SeeAct": {
        "type": "GitHub",
        "url": "https://github.com/OSU-NLP-Group/SeeAct"
    },
    "Scicode Tool Calling Agent": {
        "type": "GitHub",
        "url": "https://github.com/scicode-bench/SciCode"
    },
    "Scicode Zero Shot Agent": {
        "type": "GitHub", 
        "url": "https://github.com/scicode-bench/SciCode"
    },
    "SAB Self-Debug": {
        "type": "Paper",
        "url": "https://arxiv.org/abs/2410.05080"
    },
    "SWE-Agent": {
        "type": "GitHub",
        "url": "https://github.com/SWE-agent/SWE-agent"
    },
    "TAU-bench Few Shot": {
        "type": "GitHub",
        "url": "https://github.com/princeton-pli/hal-harness/blob/main/agents/taubench_few_shot"
    },
    "USACO Episodic + Semantic": {
        "type": "Paper",
        "url": "https://arxiv.org/abs/2404.10952"
    }
}

def create_app():
    app = Flask(__name__, 
               static_folder='static',
               template_folder='templates')
    
    preprocessor = TracePreprocessor()
    
    @app.route('/')
    def index():
        total_agents = preprocessor.get_total_agents()
        total_evaluations = preprocessor.get_total_evaluations()
        total_benchmarks = preprocessor.get_total_benchmarks()

        # Get highlight results organized by benchmark and agent
        highlights = preprocessor.get_highlight_results(limit_per_benchmark=3)

        return render_template('index.html', 
                             total_agents=total_agents, 
                             total_evaluations=total_evaluations,
                             total_benchmarks=total_benchmarks,
                             authors=AUTHORS,
                             contributors=CONTRIBUTORS,
                             highlights=highlights)
    
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}, 200
    
    @app.route("/missing")
    def missing_runs_heatmap():
        db = TracePreprocessor()
        df = db.get_all_runs()  # Should return DataFrame with columns: benchmark_name, model_name, agent_name, run_id
        fig = create_missing_runs_heatmap(df)
        heatmap_json =  json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template("missing_runs_heatmap.html", heatmap_json=heatmap_json)


    @app.route('/scienceagentbench')
    def scienceagentbench():
        models = preprocessor.get_models_for_benchmark('scienceagentbench')
        pricing = {model: DEFAULT_PRICING[model] for model in models if model in DEFAULT_PRICING}

        results_df = preprocessor.get_parsed_results_with_costs('scienceagentbench')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='scienceagentbench')

        scatter_plot, cutoff_applied = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"],
            cost_cutoff_multiplier=10,
            return_cutoff_applied=True
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Only create full plot if cost cutoff was applied
        if cutoff_applied:
            scatter_plot_full = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
            scatter_plot_full_json = json.dumps(scatter_plot_full, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            scatter_plot_full_json = None

        # heatmap = create_task_success_heatmap(
        #     preprocessor.get_task_success_data('scienceagentbench'),
        #     'ScienceAgentBench'
        # )
        # heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('scienceagentbench')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Create timeline chart
        timeline_chart = create_model_timeline_chart(leaderboard_df, 'scienceagentbench')
        timeline_chart_json = json.dumps(timeline_chart, cls=plotly.utils.PlotlyJSONEncoder)

        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        return render_template(
            'scienceagentbench.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            scatter_plot_full=scatter_plot_full_json,
            # heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='scienceagentbench',
            completion_tokens_bar=completion_tokens_json,
            timeline_chart=timeline_chart_json,
            cutoff_applied=cutoff_applied
        )

    @app.route('/usaco')
    def usaco():
        # Get models used in USACO benchmark
        usaco_models = preprocessor.get_models_for_benchmark('usaco')
        
        # Filter pricing to only show models used in USACO
        pricing = {model: DEFAULT_PRICING[model] for model in usaco_models if model in DEFAULT_PRICING}
        
        # Get data for USACO
        results_df = preprocessor.get_parsed_results_with_costs('usaco')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, benchmark_name='usaco')
        
        
        # Create scatter plot with cost cutoff and check if cutoff was applied
        scatter_plot, cutoff_applied = create_scatter_plot(
            results_df,  # Use the same data as leaderboard (with recalculated costs)
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"],
            cost_cutoff_multiplier=10,
            return_cutoff_applied=True
        )
        
        # Convert plot to JSON for rendering
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Only create full plot if cost cutoff was applied
        if cutoff_applied:
            scatter_plot_full = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
            scatter_plot_full_json = json.dumps(scatter_plot_full, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            scatter_plot_full_json = None
        
        # Create heatmap
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('usaco'),
            'USACO'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('usaco')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Create timeline chart
        timeline_chart = create_model_timeline_chart(leaderboard_df, 'usaco')
        timeline_chart_json = json.dumps(timeline_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'usaco.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            scatter_plot_full=scatter_plot_full_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='usaco',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json,
            timeline_chart=timeline_chart_json,
            cutoff_applied=cutoff_applied
        )

    @app.route('/assistantbench')
    def assistantbench():
        # Get models used in assistantbench benchmark
        assistantbench_models = preprocessor.get_models_for_benchmark('assistantbench')
        
        # Filter pricing to only show models used in assistantbench
        pricing = {model: DEFAULT_PRICING[model] for model in assistantbench_models if model in DEFAULT_PRICING}
        
        # Get data for assistantbench
        results_df = preprocessor.get_parsed_results_with_costs('assistantbench')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, benchmark_name='assistantbench')
        
        # Create scatter plot with cost cutoff and check if cutoff was applied
        scatter_plot, cutoff_applied = create_scatter_plot(
            results_df,  # Use the same data as leaderboard (with recalculated costs)
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"],
            cost_cutoff_multiplier=10.0,  # Cut off at 10x the most accurate agent's cost
            return_cutoff_applied=True
        )
        
        # Create full scatter plot (without cutoff) for toggle functionality only if cutoff was applied
        scatter_plot_full = None
        if cutoff_applied:
            scatter_plot_full = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
        
        # Convert plots to JSON for rendering
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        scatter_plot_full_json = json.dumps(scatter_plot_full, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create heatmap
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('assistantbench'),
            'AssistantBench'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('assistantbench')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Create timeline chart
        timeline_chart = create_model_timeline_chart(leaderboard_df, 'assistantbench')
        timeline_chart_json = json.dumps(timeline_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'assistantbench.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            scatter_plot_full=scatter_plot_full_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='assistantbench',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json,
            timeline_chart=timeline_chart_json,
            cutoff_applied=cutoff_applied
        )
    
    @app.route('/update_pricing/<benchmark>', methods=['POST'])
    def update_pricing(benchmark):
        pricing = request.json
        
        if benchmark == 'agentharm':
            # Get updated results with new pricing for both benchmarks
            results_df = preprocessor.get_parsed_results_with_costs('agentharm', pricing)
            results_df_benign = preprocessor.get_parsed_results_with_costs('agentharm_benign', pricing)
            
            # Create updated leaderboards
            leaderboard_df = create_leaderboard(results_df, benchmark_name='agentharm')
            leaderboard_df_benign = create_leaderboard(results_df_benign, benchmark_name='agentharm_benign')
            
            # Create updated scatter plots
            scatter_plot = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
            scatter_plot_benign = create_scatter_plot(
                results_df_benign,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
            
            # Convert scatter plots to JSON
            scatter_plot_json = json.loads(json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder))
            scatter_plot_benign_json = json.loads(json.dumps(scatter_plot_benign, cls=plotly.utils.PlotlyJSONEncoder))
            
            return jsonify({
                'leaderboard': leaderboard_df.to_dict('records'),
                'leaderboard_benign': leaderboard_df_benign.to_dict('records'),
                'scatter_plot': scatter_plot_json,
                'scatter_plot_benign': scatter_plot_benign_json
            })
        else:
            # Get updated results with new pricing
            results_df = preprocessor.get_parsed_results_with_costs(benchmark, pricing)
                        
            # Create updated leaderboard
            leaderboard_df = create_leaderboard(results_df, benchmark_name=benchmark)
            
            # Create updated scatter plot
            scatter_plot = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
            
            # Convert scatter plot to JSON
            scatter_plot_json = json.loads(json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder))
            
            return jsonify({
                'leaderboard': leaderboard_df.to_dict('records'),
                'scatter_plot': scatter_plot_json
            })

    @app.route('/corebench_hard')
    def corebench_hard():
        corebench_models = preprocessor.get_models_for_benchmark('corebench_hard')
        pricing = {model: DEFAULT_PRICING[model] for model in corebench_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('corebench_hard')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='corebench_hard')
        
        # Create scatter plot with cost cutoff and check if cutoff was applied
        scatter_plot, cutoff_applied = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"],
            cost_cutoff_multiplier=10,
            return_cutoff_applied=True
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create full scatter plot (without cutoff) for toggle functionality only if cutoff was applied
        scatter_plot_full = None
        if cutoff_applied:
            scatter_plot_full = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
        scatter_plot_full_json = json.dumps(scatter_plot_full, cls=plotly.utils.PlotlyJSONEncoder) if scatter_plot_full else None
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('corebench_hard'),
            'CORE-Bench-Hard'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('corebench_hard')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Create timeline chart
        timeline_chart = create_model_timeline_chart(leaderboard_df, 'corebench_hard')
        timeline_chart_json = json.dumps(timeline_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'corebench.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            scatter_plot_full=scatter_plot_full_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            difficulty="Hard",
            benchmark_name='corebench_hard',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json,
            timeline_chart=timeline_chart_json,
            cutoff_applied=cutoff_applied
        )

    @app.route('/gaia')
    def gaia():
        gaia_models = preprocessor.get_models_for_benchmark('gaia')
        pricing = {model: DEFAULT_PRICING[model] for model in gaia_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('gaia')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='gaia')
        
        # Create scatter plot with cost cutoff and check if cutoff was applied
        scatter_plot, cutoff_applied = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"],
            cost_cutoff_multiplier=10.0,
            return_cutoff_applied=True
        )
        
        # Create full scatter plot (without cutoff) for toggle functionality only if cutoff was applied
        scatter_plot_full = None
        if cutoff_applied:
            scatter_plot_full = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
        
        # Convert plots to JSON for rendering
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        scatter_plot_full_json = json.dumps(scatter_plot_full, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('gaia'),
            'GAIA'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('gaia')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Create timeline chart
        timeline_chart = create_model_timeline_chart(leaderboard_df, 'gaia')
        timeline_chart_json = json.dumps(timeline_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'gaia.html',  # Will need to create this template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            scatter_plot_full=scatter_plot_full_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='gaia',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json,
            timeline_chart=timeline_chart_json,
            cutoff_applied=cutoff_applied
        )

    @app.route('/taubench_airline')
    def taubench_airline():
        # Get models used in TAU-bench Airline benchmark
        taubench_airline_models = preprocessor.get_models_for_benchmark('taubench_airline')
        pricing = {model: DEFAULT_PRICING[model] for model in taubench_airline_models if model in DEFAULT_PRICING}
        
        # Get data for TAU-bench Airline
        results_df = preprocessor.get_parsed_results_with_costs('taubench_airline')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, benchmark_name='taubench_airline')
        
        # Create scatter plot with cost cutoff and check if cutoff was applied
        scatter_plot, cutoff_applied = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"],
            cost_cutoff_multiplier=10,
            return_cutoff_applied=True
        )
        
        # Convert plot to JSON for rendering
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Only create full plot if cost cutoff was applied
        if cutoff_applied:
            scatter_plot_full = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
            scatter_plot_full_json = json.dumps(scatter_plot_full, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            scatter_plot_full_json = None
        
        # Create heatmap
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('taubench_airline'),
            'TAU-bench Airline'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('taubench_airline')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Create timeline chart
        timeline_chart = create_model_timeline_chart(leaderboard_df, 'taubench_airline')
        timeline_chart_json = json.dumps(timeline_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'taubench_airline.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            scatter_plot_full=scatter_plot_full_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='taubench_airline',
            completion_tokens_bar=completion_tokens_json,
            timeline_chart=timeline_chart_json,
            cutoff_applied=cutoff_applied
        )

    @app.route('/swebench_verified_mini')
    def swebench_verified_mini():
        swebench_models = preprocessor.get_models_for_benchmark('swebench_verified_mini')
        pricing = {model: DEFAULT_PRICING[model] for model in swebench_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('swebench_verified_mini')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='swebench_verified_mini')
        
        # Create scatter plot with cost cutoff and check if cutoff was applied
        scatter_plot, cutoff_applied = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"],
            cost_cutoff_multiplier=10,
            return_cutoff_applied=True
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create full scatter plot (without cutoff) for toggle functionality only if cutoff was applied
        scatter_plot_full = None
        if cutoff_applied:
            scatter_plot_full = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
        scatter_plot_full_json = json.dumps(scatter_plot_full, cls=plotly.utils.PlotlyJSONEncoder) if scatter_plot_full else None
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('swebench_verified_mini'),
            'SWE-bench Verified (Mini)'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('swebench_verified_mini')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Create timeline chart
        timeline_chart = create_model_timeline_chart(leaderboard_df, 'swebench_verified_mini')
        timeline_chart_json = json.dumps(timeline_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'swebench_verified_mini.html',  # Use the new template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            scatter_plot_full=scatter_plot_full_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='swebench_verified_mini',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json,
            timeline_chart=timeline_chart_json,
            cutoff_applied=cutoff_applied
        )
    
    @app.route('/online_mind2web')
    def online_mind2web():
        online_mind2web_models = preprocessor.get_models_for_benchmark('online_mind2web')
        pricing = {model: DEFAULT_PRICING[model] for model in online_mind2web_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('online_mind2web')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='online_mind2web')

        completion_tokens_fig = create_completion_tokens_bar_chart('online_mind2web')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create scatter plot with cost cutoff and check if cutoff was applied
        scatter_plot, cutoff_applied = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"],
            cost_cutoff_multiplier=10,
            return_cutoff_applied=True
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create full scatter plot (without cutoff) for toggle functionality only if cutoff was applied
        scatter_plot_full = None
        if cutoff_applied:
            scatter_plot_full = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
        scatter_plot_full_json = json.dumps(scatter_plot_full, cls=plotly.utils.PlotlyJSONEncoder) if scatter_plot_full else None
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('online_mind2web'),
            'Online Mind2Web'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        # Create timeline chart
        timeline_chart = create_model_timeline_chart(leaderboard_df, 'online_mind2web')
        timeline_chart_json = json.dumps(timeline_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'online_mind2web.html',  # Use the new template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            scatter_plot_full=scatter_plot_full_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='online_mind2web',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json,
            timeline_chart=timeline_chart_json,
            cutoff_applied=cutoff_applied
        )
    
    @app.route('/scicode')
    def scicode():
        scicode_models = preprocessor.get_models_for_benchmark('scicode')
        pricing = {model: DEFAULT_PRICING[model] for model in scicode_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('scicode')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='scicode')
        
        # Create scatter plot with cost cutoff and check if cutoff was applied
        scatter_plot, cutoff_applied = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"],
            cost_cutoff_multiplier=10,
            return_cutoff_applied=True
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create full scatter plot (without cutoff) for toggle functionality only if cutoff was applied
        scatter_plot_full = None
        if cutoff_applied:
            scatter_plot_full = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
        scatter_plot_full_json = json.dumps(scatter_plot_full, cls=plotly.utils.PlotlyJSONEncoder) if scatter_plot_full else None
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('scicode'),
            'Scicode'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('scicode')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Create timeline chart
        timeline_chart = create_model_timeline_chart(leaderboard_df, 'scicode')
        timeline_chart_json = json.dumps(timeline_chart, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'scicode.html',  # Use the new template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            scatter_plot_full=scatter_plot_full_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='scicode',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json,
            timeline_chart=timeline_chart_json,
            cutoff_applied=cutoff_applied
        )


    @app.route('/failure_report/<benchmark>')
    def failure_report(benchmark):
        agent_name = request.args.get('agent')
        if not agent_name:
            return jsonify({'error': 'Agent name is required'}), 400
            
        # Get failure report for the agent
        failure_report = preprocessor.get_failure_report(agent_name, benchmark)
        if not failure_report:
            return jsonify({
                'failure_categories': [],
                'chart_data': None
            })
            
        # Create bar chart data
        categories = []
        counts = []
        for category in failure_report['failure_categories']:
            categories.append(category['category_name'])
            # Count tasks in this category
            count = sum(1 for _, classification in failure_report['task_classifications'].items() 
                       if classification['category_id'] == str(len(categories)))
            counts.append(count)
            
        chart = create_bar_chart(
            categories,
            counts,
            "Number of Tasks",
            "Failure Categories",
            "Distribution of Failure Categories"
        )
        
        return jsonify({
            'failure_categories': failure_report['failure_categories'],
            'chart_data': json.loads(json.dumps(chart, cls=plotly.utils.PlotlyJSONEncoder))
        })

    @app.route('/available_agents/<benchmark>')
    def available_agents(benchmark):
        # Get all agents for the benchmark
        agents = preprocessor.get_all_agents(benchmark)
        
        # Filter to only agents with failure reports
        agents_with_reports = [
            agent for agent in agents 
            if preprocessor.get_failure_report(agent, benchmark) is not None
        ]
        
        return jsonify(agents_with_reports)

    @app.route('/creators')
    def creators():
        return render_template('creators.html', authors=AUTHORS, contributors=CONTRIBUTORS)

    @app.route('/about')
    def about():
        return render_template('about.html')

    @app.route('/model/<model_name>')
    def model_page(model_name):
        """Model detail page showing performance across benchmarks"""
        # Decode URL-encoded characters
        model_name = unquote(model_name)
        
        try:
            # Get leaderboards for all benchmarks (excluding specified ones)
            excluded_benchmarks = ['colbench_backend_programming', 'colbench_frontend_design']
            all_leaderboards = []
            
            # Get available benchmarks by iterating through db files
            available_benchmarks = []
            for db_file in preprocessor.db_dir.glob('*.db'):
                benchmark_name = db_file.stem
                if benchmark_name not in excluded_benchmarks:
                    available_benchmarks.append(benchmark_name)
            
            for benchmark in available_benchmarks:
                try:
                    full_benchmark_df = preprocessor.get_parsed_results_with_costs(benchmark, aggregate=False)
                    if not full_benchmark_df.empty:
                        leaderboard = create_leaderboard(full_benchmark_df, benchmark)
                        # Add benchmark column for aggregation
                        leaderboard['benchmark_name'] = benchmark
                        all_leaderboards.append(leaderboard)
                except Exception as e:
                    print(f"Error creating leaderboard for {benchmark}: {e}")
                    continue
            
            if not all_leaderboards:
                return render_template('error.html', message=f"No leaderboard data found for model '{model_name}'")
            
            # Aggregate all leaderboards (with empty check to avoid future warning)
            non_empty_leaderboards = [lb for lb in all_leaderboards if not lb.empty]
            if not non_empty_leaderboards:
                return render_template('error.html', message=f"No valid leaderboard data found for model '{model_name}'")
            
            aggregated_data = pd.concat(non_empty_leaderboards, ignore_index=True)
            
            # Filter for the specific model
            model_data = aggregated_data[aggregated_data['Models'] == model_name]
            
            if model_data.empty:
                return render_template('error.html', message=f"No data found for model '{model_name}'")

            # Calculate model info
            model_info = {
                'name': model_name,
                'total_benchmarks': model_data['benchmark_name'].nunique(),
                'total_agents': model_data['Agent Name'].nunique(),
                'avg_accuracy': model_data['Accuracy'].mean(),
                'avg_cost': model_data['Total Cost'].mean(),
                'first_seen': model_data['Date'].min() if 'Date' in model_data.columns and not model_data['Date'].isna().all() else 'Unknown',
                'last_seen': model_data['Date'].max() if 'Date' in model_data.columns and not model_data['Date'].isna().all() else 'Unknown',
                'pricing': preprocessor.get_model_pricing(model_name)
            }

            # Prepare benchmark performance data
            benchmark_performance = []
            for _, row in model_data.iterrows():
                benchmark_performance.append({
                    'benchmark': row['benchmark_name'],
                    'benchmark_title': row['benchmark_name'].replace('_', ' ').title(),
                    'accuracy': row['Accuracy'],
                    'cost': row['Total Cost'],
                    'cost_display': f"${row['Total Cost']:.2f}",
                    'on_frontier': row.get('Is Pareto', False),
                    'agent_name': row['Agent Name']
                })

            # Sort by benchmark name alphabetically
            benchmark_performance.sort(key=lambda x: x['benchmark_title'])
            
            return render_template('model_page.html', 
                                 model_info=model_info, 
                                 benchmark_performance=benchmark_performance)
                                 
        except Exception as e:
            print(f"Error getting model data: {e}")
            return render_template('error.html', message=f"Error loading data for model '{model_name}'")

    @app.route('/agent/<agent_name>')
    def agent_page(agent_name):
        """Agent detail page showing performance across benchmarks"""
        agent_name = unquote(agent_name)
        base_agent_name = agent_name.split(' (')[0] if ' (' in agent_name else agent_name

        try:
            # Get leaderboards for all benchmarks (excluding specified ones)
            excluded_benchmarks = ['colbench_backend_programming', 'colbench_frontend_design']
            all_leaderboards = []
            
            # Get available benchmarks by iterating through db files
            available_benchmarks = []
            for db_file in preprocessor.db_dir.glob('*.db'):
                benchmark_name = db_file.stem
                if benchmark_name not in excluded_benchmarks:
                    available_benchmarks.append(benchmark_name)
            
            for benchmark in available_benchmarks:
                try:
                    full_benchmark_df = preprocessor.get_parsed_results_with_costs(benchmark, aggregate=False)
                    if not full_benchmark_df.empty:
                        leaderboard = create_leaderboard(full_benchmark_df, benchmark)
                        # Add benchmark column for aggregation
                        leaderboard['benchmark_name'] = benchmark
                        all_leaderboards.append(leaderboard)
                except Exception as e:
                    print(f"Error creating leaderboard for {benchmark}: {e}")
                    continue
            
            if not all_leaderboards:
                return render_template('error.html', message=f"No leaderboard data found for agent '{agent_name}'")
            
            # Aggregate all leaderboards (with empty check to avoid future warning)
            non_empty_leaderboards = [lb for lb in all_leaderboards if not lb.empty]
            if not non_empty_leaderboards:
                return render_template('error.html', message=f"No valid leaderboard data found for agent '{agent_name}'")
            
            aggregated_data = pd.concat(non_empty_leaderboards, ignore_index=True)
            
            # Filter for the specific agent
            agent_data = aggregated_data[aggregated_data['Agent Name'] == base_agent_name]
            
            if agent_data.empty:
                return render_template('error.html', message=f"No data found for agent '{agent_name}'")

            # Calculate agent info
            agent_link = AGENT_LINKS.get(base_agent_name, None)
            agent_info = {
                'name': agent_name,
                'models': list(agent_data['Models'].unique()) if 'Models' in agent_data.columns else [],
                'total_benchmarks': agent_data['benchmark_name'].nunique(),
                'first_seen': agent_data['Date'].min() if 'Date' in agent_data.columns and not agent_data['Date'].isna().all() else 'Unknown',
                'last_seen': agent_data['Date'].max() if 'Date' in agent_data.columns and not agent_data['Date'].isna().all() else 'Unknown',
                'total_runs': len(agent_data),
                'pareto_benchmarks': len(agent_data[agent_data.get('Is Pareto', False) == True]),
                'link': agent_link
            }

            # Prepare benchmark performance data
            benchmark_performance = []
            for _, row in agent_data.iterrows():
                benchmark_performance.append({
                    'benchmark': row['benchmark_name'],
                    'benchmark_title': row['benchmark_name'].replace('_', ' ').title(),
                    'accuracy': row['Accuracy'],
                    'cost': row['Total Cost'],
                    'on_frontier': row.get('Is Pareto', False),
                    'model_name': row.get('Models', 'Unknown')
                })

            # Sort by benchmark name alphabetically
            benchmark_performance.sort(key=lambda x: x['benchmark_title'])
            
            return render_template('agent_page.html', agent_info=agent_info, benchmark_performance=benchmark_performance)

        except Exception as e:
            print(f"Error getting agent data: {e}")
            return render_template('error.html', message=f"Error loading data for agent '{agent_name}'")

    @app.route('/insights')
    def insights():
        # Twitter embed HTML - these are actual tweets from @halevals and related accounts
        twitter_embeds = [
            '''<blockquote class="twitter-tweet"><p lang="en" dir="ltr">How does GPT-5 compare against Claude Opus 4.1 on agentic tasks? <br><br>Since their release, we have been evaluating these models on challenging science, web, service, and code tasks. <br><br>Headline result: While cost-effective, so far GPT-5 never tops agentic leaderboards. More evals 🧵 <a href="https://t.co/KhVcXZN3fc">pic.twitter.com/KhVcXZN3fc</a></p>&mdash; Sayash Kapoor (@sayashk) <a href="https://twitter.com/sayashk/status/1953952498105946594?ref_src=twsrc%5Etfw">August 8, 2025</a></blockquote>''',
            '''<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GPT-OSS underperforms even on benchmarks that require raw tool calling. For example, CORE-Bench requires agents to run bash commands to reproduce scientific papers. <br><br>DeepSeek V3 scores 18%. <br>GPT-OSS scores 11%.<a href="https://t.co/EVxxSqKFMe">https://t.co/EVxxSqKFMe</a> <a href="https://t.co/tx8rTygUWw">pic.twitter.com/tx8rTygUWw</a></p>&mdash; Sayash Kapoor (@sayashk) <a href="https://twitter.com/sayashk/status/1955298275923210389?ref_src=twsrc%5Etfw">August 12, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>''',
            '''<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can AI agents reliably navigate the web? Does the choice of agent scaffold affect web browsing ability? To answer these questions, we added Online Mind2Web, a web browsing benchmark, to the Holistic Agent Leaderboard (HAL). <br><br>We evaluated 9 models (including GPT-5 and Sonnet 4)… <a href="https://t.co/jwS2iFG27E">pic.twitter.com/jwS2iFG27E</a></p>&mdash; Sayash Kapoor (@sayashk) <a href="https://twitter.com/sayashk/status/1963343022252315112?ref_src=twsrc%5Etfw">September 3, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>''',
            '''<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We have added ScienceAgentBench to HAL and evaluated it with leading models (GPT-5, o3, Opus 4.1). <br><br>o3 tops the leaderboard at a lower cost than GPT-5, Opus 4.1, and Sonnet 3.7 High. o4-mini Low is much cheaper than the crowd, but with similar accuracy.<br><br>Grateful to so many… <a href="https://t.co/Xw6iWhqDoe">https://t.co/Xw6iWhqDoe</a> <a href="https://t.co/1ZNXaLK7kz">pic.twitter.com/1ZNXaLK7kz</a></p>&mdash; Sayash Kapoor (@sayashk) <a href="https://twitter.com/sayashk/status/1966144670561612202?ref_src=twsrc%5Etfw">September 11, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>''',
            '''<blockquote class="twitter-tweet"><p lang="en" dir="ltr">OpenAI claims hallucinations persist because evaluations reward guessing and that GPT-5 is better calibrated. Do results from HAL support this conclusion? On AssistantBench, a general web search benchmark, GPT-5 has higher precision and lower guess rates than o3! <a href="https://t.co/HxGgVLkIyN">pic.twitter.com/HxGgVLkIyN</a></p>&mdash; Peter Kirgis (@PKirgis) <a href="https://twitter.com/PKirgis/status/1966547382033936577?ref_src=twsrc%5Etfw">September 12, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>''',
            '''<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We spent the last year evaluating agents for HAL.<br><br>My biggest learning: We live in the Windows 95 era of agent evaluation. <a href="https://t.co/DeIzWm1f0c">pic.twitter.com/DeIzWm1f0c</a></p>&mdash; Sayash Kapoor (@sayashk) <a href="https://twitter.com/sayashk/status/1967998405852152039?ref_src=twsrc%5Etfw">September 16, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>''',
            '''<blockquote class="twitter-tweet"><p lang="en" dir="ltr">CORE-Bench is solved (using Opus 4.5 with Claude Code)<br><br>TL;DR: Last week, we released results for Opus 4.5 on CORE-Bench, a benchmark that tests agents on scientific reproducibility tasks. Earlier this week, Nicholas Carlini reached out to share that an updated scaffold that uses… <a href="https://t.co/vF2y4Fbe40">pic.twitter.com/vF2y4Fbe40</a></p>&mdash; Sayash Kapoor (@sayashk) <a href="https://twitter.com/sayashk/status/1996334941832089732?ref_src=twsrc%5Etfw">December 3, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>'''
        ]

        twitter_embeds.reverse()

        return render_template('insights.html', twitter_embeds=twitter_embeds)

    @app.route('/press')
    def press():
        # News articles data
        news_articles = [
            {
                'title': 'Is AI hitting a wall?',
                'url': 'https://www.ft.com/content/d01290c9-cc92-4c1f-bd70-ac332cd40f94',
                'source': 'Financial Times',
                'date': 'August 2025'
            },
            {
                'title': 'Developers Say GPT-5 Is a Mixed Bag',
                'url': 'https://www.wired.com/story/gpt-5-coding-review-software-engineering/',
                'source': 'WIRED',
                'date': 'August 2025'
            }
        ]
        
        return render_template('press.html', news_articles=news_articles)

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=False, port=5001, host='0.0.0.0')