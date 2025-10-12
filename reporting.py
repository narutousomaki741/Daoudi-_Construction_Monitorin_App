import tempfile
import os
import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger
import plotly.express as px
import pandas as pd
import json
import html
import logging


DEFAULT_DISCIPLINE_COLORS = {
    "Préliminaires": "#FF6B6B",
    "Terrassement": "#4ECDC4",
    "Fondations": "#45B7D1",
    "GrosOeuvres": "#96CEB4",
    "SecondOeuvres": "#FECA57",
    "default": "#BDC3C7"
}

def _validate_required_columns(df: pd.DataFrame, required: set, name: str = "DataFrame") -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(list(missing))}")
def _convert_schedule_dates(df: pd.DataFrame, start_col: str = 'Start', end_col: str = 'End') -> pd.DataFrame:
    df = df.copy()
    if start_col not in df.columns or end_col not in df.columns:
        # accept alternative column name 'Finish'
        if 'Finish' in df.columns and end_col not in df.columns:
            end_col = 'Finish'
    df['Start'] = pd.to_datetime(df[start_col], errors='coerce')
    df['End'] = pd.to_datetime(df[end_col], errors='coerce')
    if df['Start'].isna().any() or df['End'].isna().any():
        raise ValueError("Some Start or End values could not be parsed to datetime in schedule.")
    return df

def generate_interactive_gantt(schedule_df: pd.DataFrame, output_path: str) -> str:
    """
    Enhanced Interactive Gantt Chart for large datasets:
    - Dynamic height based on visible tasks
    - Fixed or responsive width
    - Filters by Discipline & Zone
    - Task selection via legend
    - Jump-to-task & timeline fit
    - Split & Download PNGs
    """
    
    logger = logging.getLogger(__name__)

    required = {"TaskID", "Discipline", "Start", "End"}
    _validate_required_columns(schedule_df, required, "schedule_df")

    df = _convert_schedule_dates(schedule_df).copy()
    df['TaskID'] = df['TaskID'].astype(str)
    df['TaskName'] = df.get('TaskName', df['TaskID'])
    df['TaskID_Legend'] = df['TaskID'].apply(lambda x: x.split('-')[0])
    df['TaskZone'] = df.get('Zone', df['TaskID'].apply(lambda x: x.split('-')[-1] if '-' in x else ''))

    df['DisplayName'] = df.apply(lambda r: f"{r['TaskName']} {'-'.join(r['TaskID'].split('-')[1:]) if '-' in r['TaskID'] else r['TaskID']}", axis=1)
    df = df.sort_values(['Start','Discipline','TaskZone','TaskID_Legend'])
    df['DurationDays'] = (df['End'] - df['Start']).dt.total_seconds() / (3600 * 24)

    # Color mapping
    unique_disc = df['Discipline'].astype(str).unique().tolist()
    color_discrete_map = {}
    fallback_colors = px.colors.qualitative.Plotly
    for i, d in enumerate(unique_disc):
        color_discrete_map[d] = DEFAULT_DISCIPLINE_COLORS.get(d, fallback_colors[i % len(fallback_colors)])

    traces_data = []
    trace_meta = []
    all_tasks_data = []
    task_index_map = {name: i for i, name in enumerate(df['DisplayName'].tolist())}
    for _, row in df.iterrows():
        discipline = str(row['Discipline'])
        zone = str(row['TaskZone'])
        task_id = str(row['TaskID'])
        task_name = str(row['TaskName'])
        display_name = str(row['DisplayName'])
        start_date = row['Start'].strftime('%Y-%m-%d')
        end_date = row['End'].strftime('%Y-%m-%d')
        duration = float(row['DurationDays'])
        color = str(color_discrete_map.get(discipline, 'blue'))
        if end_date==start_date:
            end_date=pd.to_datetime(end_date)+pd.Timedelta(days=0.3)
            end_date=end_date.strftime('%Y-%m-%d %H:%M:%S')
            trace = {
            'x': [start_date, end_date],
            'y': [display_name, display_name],
            'mode': 'lines',
            'line': {'color': color, 'width': 8},
            'name': f"{discipline} | {zone}" if zone else discipline,
            'hovertemplate': (
                f"<b>{html.escape(task_name)}</b><br>"
                f"ID: {html.escape(task_id)}<br>"
                f"Start: {start_date}<br>"
                f"End: {end_date}<br>"
                f"Duration: {duration:.1f} days<extra></extra>"
            ),
            'showlegend': False
             }
        else:
            trace = {
            'x': [start_date, end_date],
            'y': [display_name, display_name],
            'mode': 'lines',
            'line': {'color': color, 'width': 8},
            'name': f"{discipline} | {zone}" if zone else discipline,
            'hovertemplate': (
                f"<b>{html.escape(task_name)}</b><br>"
                f"ID: {html.escape(task_id)}<br>"
                f"Start: {start_date}<br>"
                f"End: {end_date}<br>"
                f"Duration: {duration:.1f} days<extra></extra>"
            ),
            'showlegend': False
             }
        traces_data.append(trace)
        trace_meta.append({
            'trace_index': len(traces_data)-1,
            'discipline': discipline,
            'zone': zone,
            'task_id': task_id,
            'display_name': display_name,
            'task_name': task_name,
            'selected': True
        })
        all_tasks_data.append({
            'TaskID': task_id,
            'TaskName': task_name,
            'DisplayName': display_name,
            'Discipline': discipline,
            'Zone': zone
        })
    all_dates = []
    for t in traces_data:
        all_dates.extend([pd.to_datetime(t['x'][0]), pd.to_datetime(t['x'][1])])
    min_date = min(all_dates)
    max_date = max(all_dates)
    week_lines = []
    current_week = min_date
    week_num = 1
    while current_week <= max_date:
        week_lines.append(current_week.strftime('%Y-%m-%d'))
        current_week += pd.Timedelta(weeks=1)
        week_num += 1
    # Initial layout
    layout_data = {
        'title': {'text': 'Interactive Project Schedule Gantt',
                 'x': 0.5,
                 'xanchor': 'center',
                 'y': 0.99,           # ⬅ how high above the top edge of the plot area
                  'yref': 'paper',     # ⬅ relative to the full figure, not data
                  'font': {'size': 24,   'family': 'Arial Black, sans-serif','color': 'black'}
                  },

        'height': max(600, len(df)*25), 
        'margin': {'l':300, 'r':40, 't':10, 'b':120},
        'xaxis': {
            'title':'Date',
            'type':'date',
            'rangeselector':{'buttons':[
                      {'count':1,'label':'1m','step':'month','stepmode':'backward'},
                      {'count':6,'label':'6m','step':'month','stepmode':'backward'},
                      {'count':14,'label':'14m','step':'month','stepmode':'backward'},
                      {'count':30,'label':'30m','step':'month','stepmode':'backward'},
                      {'step':'all','label':'Fit'}],
                       'y': 0.98},
            'showgrid':True,
            'gridcolor':'grey',
            'griddash': 'solid',
            'gridwidth':1,  
            'dtick': 604800000,          
            'showticklabels': False 
            
        },
        'yaxis': { 'title':'Tasks','autorange':True, 'yaxis.automargin': False,
             'showgrid':True,'gridcolor':'lightgrey','gridwidth':1, 'domain': [0, 0.97]}
    }

    # JSON for JS
    traces_data_json = json.dumps(traces_data)
    layout_data_json = json.dumps(layout_data)
    trace_meta_json = json.dumps(trace_meta)
    all_tasks_json = json.dumps(all_tasks_data)

    disciplines = sorted(df['Discipline'].astype(str).unique().tolist())
    zones = sorted([z for z in df['TaskZone'].astype(str).unique().tolist() if z])

    # HTML content
    html_content = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Interactive Gantt - Split PNG</title>
<style>
body {{ font-family: Arial; margin:12px; }}
.controls {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:12px; align-items:center; }}
.plot-container {{ width:100vw; height:600px; overflow:auto; border:1px solid #ddd; border-radius:4px; }}
.task-legend-table {{ border-collapse: collapse; width: 100%; font-size:12px; max-height:400px; overflow:auto; display:block; }}
.task-legend-table th, .task-legend-table td {{ border:1px solid #ddd; padding:6px; text-align:left; }}
.task-legend-table th {{ background:#f4f4f4; position: sticky; top:0; z-index: 10; }}
.task-row {{ cursor:pointer; }}
.task-row:hover {{ background-color:#f0f8ff; }}
.task-row.selected {{ background-color:#e6f3ff; }}
.task-checkbox {{ margin:0; cursor:pointer; }}
.legend-panel {{ max-height:450px; overflow:auto; border:1px solid #eee; padding:8px; background:#fff; }}
.collapsible {{ cursor:pointer; padding:8px; background:#f7f7f7; border:1px solid #e8e8e8; margin-bottom:6px; border-radius:4px; }}
.small {{ font-size:12px; color:#555; }}
.btn {{ padding:6px 8px; border-radius:4px; border:1px solid #ccc; background:#fff; cursor:pointer; }}
.btn-primary {{ background:#007bff; color:white; border-color:#007bff; }}
input[type=text], input[type=number] {{ padding:6px; border:1px solid #ccc; border-radius:4px; width:60px; }}
.selection-controls {{ display:flex; gap:8px; margin:8px 0; flex-wrap:wrap; }}
</style>
</head>
<body>
<h2>Interactive Project Schedule</h2>
<div class="controls">
<div><label class="small">Discipline:</label>
<select id="discipline-filter" class="btn">
<option value="__all__">All</option>
{''.join(f'<option value="{html.escape(str(d))}">{html.escape(str(d))}</option>' for d in disciplines)}
</select></div>
<div><label class="small">Zone:</label>
<select id="zone-filter" class="btn">
<option value="__all__">All</option>
{''.join(f'<option value="{html.escape(str(z))}">{html.escape(str(z))}</option>' for z in zones)}
</select></div>
<div><label class="small">Jump to Task:</label>
<input id="jump-input" type="text" placeholder="e.g. 1.1_F0_A"/>
<button id="jump-btn" class="btn">Go</button></div>
<div>
<button id="toggle-legend" class="btn">Toggle Legend</button>
<button id="fit-btn" class="btn">Fit Timeline</button>
<button id="reset-view" class="btn">Reset View</button>
</div>
<div>
<label class="small">Tasks per PNG:</label><input id="tasks-per-section" type="number" value="50" min="1"/>
<button id="split-download" class="btn">Split & Download PNGs</button>
</div>
</div>

<div class="plot-container"><div id="plot"style="width:100%; height:100%;"></div></div>

<div style="margin-top:12px;">
<div class="collapsible" id="legend-toggle-header">📋 Task Selection (click to expand/collapse)</div>
<div class="legend-panel" id="legend-panel">
<div class="selection-controls">
<button id="select-all" class="btn">Select All</button>
<button id="deselect-all" class="btn">Deselect All</button>
<button id="apply-selection" class="btn btn-primary">Apply Selection</button>
<span id="selection-count" class="small" style="margin-left:auto;">{len(df)} tasks selected</span>
</div>
<table id="task-legend" class="task-legend-table">
<thead><tr><th>Show</th><th>Task ID</th><th>Task Name</th><th>Discipline</th><th>Zone</th></tr></thead>
<tbody>
{''.join(f'<tr class="task-row" data-task-id="{html.escape(str(r["TaskID"]))}" data-selected="true">\
<td><input type="checkbox" class="task-checkbox" checked data-task-id="{html.escape(str(r["TaskID"]))}"></td>\
<td>{html.escape(str(r["TaskID_Legend"]))}</td>\
<td>{html.escape(str(r["TaskName"]))}</td>\
<td>{html.escape(str(r["Discipline"]))}</td>\
<td>{html.escape(str(r["TaskZone"]))}</td></tr>' for _, r in df.iterrows())}
</tbody>
</table>
</div>
</div>

<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
<script>
const traceMeta={trace_meta_json};
const allTasks={all_tasks_json};
const allTracesData={traces_data_json};
const plotLayout={layout_data_json};

let selectedTasks = new Set(allTasks.map(t=>t.TaskID));
let currentVisibleTasks = new Set(selectedTasks);

function updateWeeklyAnnotations() {{
    if (!window.plotDiv) return;
    
    const visibleDates = [];
    allTracesData.forEach((trace, index) => {{
        if (trace.visible !== false) {{
            visibleDates.push(new Date(trace.x[0]), new Date(trace.x[1]));
        }}
    }});
    
    if (visibleDates.length === 0) return;
    
    const minDate = new Date(Math.min(...visibleDates));
    const maxDate = new Date(Math.max(...visibleDates));
    
    const weeklyAnnotations = [];
    let currentWeek = new Date(minDate);
    let weekCount = 0;
    
    currentWeek.setDate(currentWeek.getDate() - currentWeek.getDay() + 1);
    
    // Calculate approximate pixel distance between weeks for font size
    const weekMs = 7 * 24 * 60 * 60 * 1000;
    const totalWeeks = (maxDate - minDate) / weekMs;
    const chartWidth = document.querySelector('.plot-container').clientWidth;
    const pixelsPerWeek = chartWidth / totalWeeks;
    const fontSize = Math.min(17, pixelsPerWeek * 0.9); // 90% of week width, min 10px
    
    while (currentWeek <= maxDate) {{
        weekCount++;
        
        if (weekCount % 6 === 0) {{
            // Every 5th week: show date
            weeklyAnnotations.push({{
                x: currentWeek.toISOString().split('T')[0],
                y: 0,
                xref: 'x',
                yref: 'paper',
                text: currentWeek.toLocaleDateString('en-US', {{
                    day: 'numeric', 
                    month: 'short',
                    year: 'numeric' 
                }}),
                showarrow: false,
                yshift: -5,
                font: {{ size: fontSize , family: 'Arial Black'}},
                textangle: -90,
                // ↓↓↓ CENTER THE TEXT ↓↓↓
                xanchor: 'center',  // Center horizontally
                yanchor: 'top'      // Anchor at top of text
            }});
        }} else {{
            // All other weeks: show week number
            weeklyAnnotations.push({{
                x: currentWeek.toISOString().split('T')[0],
                y: 0,
                xref: 'x',
                yref: 'paper',
                text: `W${{weekCount}}`,
                showarrow: false,
                yshift: -15,
                font: {{ size: fontSize }},
                textangle: -90,
                // ↓↓↓ CENTER THE TEXT ↓↓↓
                xanchor: 'center',  // Center horizontally
                yanchor: 'top'      // Anchor at top of text
            }});
        }}
        
        currentWeek.setDate(currentWeek.getDate() + 7);
    }}
    
    Plotly.relayout(window.plotDiv, {{
        annotations: weeklyAnnotations
    }});
}}
function initializePlot(){{
    const initialCategoryArray = allTasks.map(t=>t.DisplayName);
    const initialHeight = Math.max(600, initialCategoryArray.length*25);
    const plotLayoutDynamic = {{
        ...plotLayout,
        height: initialHeight,
        yaxis: {{
            ...plotLayout.yaxis,
            categoryarray: initialCategoryArray,
            tickvals: initialCategoryArray,
            ticktext: initialCategoryArray,
            range: [-0.5, initialCategoryArray.length - 0.5] 
        }}
    }};
    Plotly.newPlot('plot', allTracesData, plotLayoutDynamic).then(plotDiv => {{
        window.plotDiv = plotDiv;
        updateWeeklyAnnotations();
        updateSelectionDisplay();
    }});
}}

function updateChartWithSelection(){{
    if(!window.plotDiv) return;

    currentVisibleTasks = new Set(selectedTasks);

    const visibleCategoryArray = allTasks
        .filter(t => currentVisibleTasks.has(t.TaskID))
        .map(t => t.DisplayName);

    if(visibleCategoryArray.length===0){{
        console.warn("No tasks to display after filtering.");
        return;
    }}

    const maxLabelLength = Math.max(...visibleCategoryArray.map(v=>v.length),10);
    const fontSize = Math.max(8, 20 - (maxLabelLength/3));
    const newHeight = Math.max(600, visibleCategoryArray.length*25);

    const visibilityUpdates = traceMeta.map(tm => currentVisibleTasks.has(tm.task_id));
    const fixedDistanceTop = 30; // pixels
    const plotAreaTopMargin = fixedDistanceTop;
    Plotly.restyle(window.plotDiv, {{visible: visibilityUpdates}}).then(()=>{{
        return Plotly.relayout(window.plotDiv, {{
            'yaxis.categoryarray': visibleCategoryArray,
            'yaxis.tickvals': visibleCategoryArray,
            'yaxis.ticktext': visibleCategoryArray,
            'yaxis.tickfont.size': fontSize,
            'yaxis.range': [-0.5, visibleCategoryArray.length - 0.5],
            'height': newHeight,
            'margin.l': 290,
            'margin.r': 40,
            'margin.t': plotAreaTopMargin,
            'margin.b': 120,
            'xaxis.autorange': true
        }});
    }}).then(()=>{{ updateWeeklyAnnotations();updateSelectionDisplay(); }});
}}

function updateSelectionDisplay(){{
    const selectedCount = selectedTasks.size;
    const visibleCount = currentVisibleTasks.size;
    document.getElementById('selection-count').textContent = `${{selectedCount}} selected, ${{visibleCount}} visible`;
    document.querySelectorAll('.task-row').forEach(row=>{{
        const taskId = row.getAttribute('data-task-id');
        const checkbox = row.querySelector('.task-checkbox');
        const isSelected = selectedTasks.has(taskId);
        const isVisible = currentVisibleTasks.has(taskId);
        checkbox.checked = isSelected;
        row.setAttribute('data-selected', isSelected);
        row.classList.toggle('selected', isSelected);
        row.style.opacity = isVisible?'1':'0.4';
    }});
}}

function toggleTaskSelection(taskId, selected){{
    if(selected) selectedTasks.add(taskId);
    else selectedTasks.delete(taskId);
}}
function selectAllTasks(){{ allTasks.forEach(t=>selectedTasks.add(t.TaskID)); updateChartWithSelection(); }}
function deselectAllTasks(){{ selectedTasks.clear(); updateChartWithSelection(); }}
function resetToFullView(){{
    selectedTasks = new Set(allTasks.map(t=>t.TaskID));
    document.getElementById('discipline-filter').value='__all__';
    document.getElementById('zone-filter').value='__all__';
    updateChartWithSelection();
    updateWeeklyAnnotations();
}}

function applyFilters(){{
    const selDisc = document.getElementById('discipline-filter').value;
    const selZone = document.getElementById('zone-filter').value;
    selectedTasks.clear();
    allTasks.forEach(t=>{{
        if((selDisc==='__all__'||t.Discipline===selDisc) && (selZone==='__all__'||t.Zone===selZone))
            selectedTasks.add(t.TaskID);
    }});
    updateChartWithSelection();
}}

function splitAndDownloadCharts(tasksPerSection=50){{
    const visibleTasksArr = Array.from(currentVisibleTasks);
    let start = 0, sectionNum = 1;

    // Get current weekly annotations for reuse
    const currentAnnotations = window.plotDiv?.layout?.annotations || [];

    function processNextSection() {{
        if (start >= visibleTasksArr.length) return;
        
        const sectionTasks = visibleTasksArr.slice(start, start + tasksPerSection);
        const sectionCategoryArray = allTasks
            .filter(t => sectionTasks.includes(t.TaskID))
            .map(t => t.DisplayName);

        if (sectionCategoryArray.length === 0) {{
            start += tasksPerSection;
            processNextSection();
            return;
        }}

        const visibilityUpdates = traceMeta.map(tm => sectionTasks.includes(tm.task_id));

        // Calculate date range for this section
        const sectionDates = [];
        traceMeta.forEach((tm, index) => {{
            if (sectionTasks.includes(tm.task_id)) {{
                const trace = allTracesData[index];
                sectionDates.push(new Date(trace.x[0]), new Date(trace.x[1]));
            }}
        }});

        if (sectionDates.length === 0) {{
            start += tasksPerSection;
            processNextSection();
            return;
        }}

        const minDate = new Date(Math.min(...sectionDates));
        const maxDate = new Date(Math.max(...sectionDates));

        // Create temporary div for this section
        const tempDiv = document.createElement('div');
        tempDiv.style.position = 'absolute';
        tempDiv.style.left = '-9999px';
        tempDiv.style.width = '1200px'; // Fixed width for consistent PNGs
        tempDiv.style.height = Math.max(600, sectionCategoryArray.length * 25) + 'px';
        document.body.appendChild(tempDiv);

        try {{
            // Create filtered traces - use boolean visibility, not 'legendonly'
            const filteredTraces = allTracesData.map((trace, index) => ({{
                ...trace,
                visible: visibilityUpdates[index]
            }}));

            const tempLayout = {{
                ...plotLayout,
                title: {{ ...plotLayout.title, 
                    text: `Gantt Section ${{sectionNum}}: ${{minDate.toLocaleDateString('en-US', {{day:'numeric', month:'short', year:'numeric'}})}} to ${{maxDate.toLocaleDateString('en-US', {{day:'numeric', month:'short', year:'numeric'}})}}`
                   }},
                yaxis: {{
                    ...plotLayout.yaxis,
                    categoryarray: sectionCategoryArray,
                    tickvals: sectionCategoryArray,
                    ticktext: sectionCategoryArray,
                    autorange: true
                }},
                xaxis: {{
                    ...plotLayout.xaxis,
                    range: [minDate.toISOString().slice(0,10), maxDate.toISOString().slice(0,10)],
                    autorange: false
                }},
                height: tempDiv.offsetHeight,
                width: tempDiv.offsetWidth,
                annotations: currentAnnotations, // Use existing annotations
                showlegend: false
            }};

            Plotly.newPlot(tempDiv, filteredTraces, tempLayout)
                .then(() => {{
                    // Wait a bit for rendering to complete
                    return new Promise(resolve => setTimeout(resolve, 500));
                }})
                .then(() => {{
                    return Plotly.toImage(tempDiv, {{
                        format: 'png',
                        height: tempDiv.offsetHeight,
                        width: tempDiv.offsetWidth
                    }});
                }})
                .then(dataUrl => {{
                    const a = document.createElement('a');
                    a.href = dataUrl;
                    a.download = `Gantt_Section_${{sectionNum}}_(${{sectionTasks.length}}_tasks).png`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    
                    // Clean up
                    Plotly.purge(tempDiv);
                    document.body.removeChild(tempDiv);
                    
                    // Process next section
                    start += tasksPerSection;
                    sectionNum++;
                    processNextSection();
                }})
                .catch(error => {{
                    console.error('Error generating PNG:', error);
                    alert(`Error generating section ${{sectionNum}}: ${{error.message}}`);
                    
                    // Clean up even on error
                    Plotly.purge(tempDiv);
                    document.body.removeChild(tempDiv);
                    
                    // Continue with next section
                    start += tasksPerSection;
                    sectionNum++;
                    processNextSection();
                }});
                
        }} catch (error) {{
            console.error('Error setting up plot:', error);
            document.body.removeChild(tempDiv);
            start += tasksPerSection;
            sectionNum++;
            processNextSection();
        }}
    }}

    // Start processing sections
    processNextSection();
}}

document.addEventListener('click', function(e){{
    if(e.target.classList.contains('task-checkbox')){{
        toggleTaskSelection(e.target.getAttribute('data-task-id'), e.target.checked);
        updateChartWithSelection();
    }}
    if(e.target.closest('.task-row') && !e.target.classList.contains('task-checkbox')){{
        const row = e.target.closest('.task-row');
        const checkbox = row.querySelector('.task-checkbox');
        const newState = !checkbox.checked;
        checkbox.checked = newState;
        toggleTaskSelection(row.getAttribute('data-task-id'), newState);
        updateChartWithSelection();
    }}
}});

document.addEventListener('DOMContentLoaded', function(){{
    initializePlot();
    document.getElementById('select-all').addEventListener('click', selectAllTasks);
    document.getElementById('deselect-all').addEventListener('click', deselectAllTasks);
    document.getElementById('apply-selection').addEventListener('click', updateChartWithSelection);
    document.getElementById('discipline-filter').addEventListener('change', applyFilters);
    document.getElementById('zone-filter').addEventListener('change', applyFilters);
    document.getElementById('reset-view').addEventListener('click', resetToFullView);
    document.getElementById('jump-btn').addEventListener('click', function(){{
        const q=document.getElementById('jump-input').value.trim().toLowerCase();
        if(!q || !window.plotDiv) return;
        const foundTask = allTasks.find(t => t.TaskID.toLowerCase().includes(q) || t.TaskName.toLowerCase().includes(q));
        if(foundTask){{
            const layout = window.plotDiv.layout || {{}};
            const yaxis = layout.yaxis || {{}};
            const currentCategoryArray = yaxis.categoryarray || [];
            const idx = currentCategoryArray.indexOf(foundTask.DisplayName);
            if(idx>=0){{
                const visibleCount = 10;
                const startIdx = Math.max(0, idx - Math.floor(visibleCount/2));
                const endIdx = Math.min(currentCategoryArray.length-1, startIdx + visibleCount -1);
                Plotly.relayout(window.plotDiv, {{'yaxis.range':[endIdx, startIdx]}});
            }} else {{ alert('Task found but not in current view. Try resetting filters.'); }}
        }} else {{ alert('Task not found: '+q); }}
    }});
    document.getElementById('toggle-legend').addEventListener('click', ()=>{{ const panel=document.getElementById('legend-panel'); panel.style.display=(panel.style.display==='none')?'block':'none'; }});
    document.getElementById('fit-btn').addEventListener('click', ()=>{{ if(window.plotDiv) Plotly.relayout(window.plotDiv, {{'xaxis.autorange':true,'yaxis.autorange':true}}); }});
    document.getElementById('split-download').addEventListener('click', ()=>{{
        const tps = parseInt(document.getElementById('tasks-per-section').value) || 50;
        splitAndDownloadCharts(tps);
    }});
}});
</script>
</body>
</html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logger.info(f"Enhanced interactive HTML Gantt saved: {output_path}")
    return output_path

class BasicReporter:
    def __init__(self, tasks: List, schedule: Dict[str, tuple],
                 worker_manager, equipment_manager, calendar):
        self.tasks = tasks
        self.schedule = schedule
        self.worker_manager = worker_manager
        self.equipment_manager = equipment_manager
        self.calendar = calendar
    
    # ---------------------------------------------------------
    # 1️⃣ Export basic schedule
    # ---------------------------------------------------------
    def export_schedule(self, path):
        rows = []
        for t in self.tasks:
            if t.id in self.schedule:
                s, e = self.schedule[t.id]
                rows.append({
                    "TaskID": t.id,
                    "TaskName": t.name,
                    "Discipline": t.discipline,
                    "Zone": t.zone,
                    "Floor": t.floor,
                    "Start": pd.to_datetime(s).date(),
                    "End": pd.to_datetime(e).date(),
                    "ResourceType": t.resource_type,
                    "CrewsNeeded": t.allocated_crews if t.task_type in ("worker", "hybrid") else "",
                    "EquipmentNeeded": ", ".join([f"{k} x{v}" for k, v in (t.allocated_equipments or {}).items()]),
                    "Quantity": t.quantity,
                    "TaskType": t.task_type,
                })
        pd.DataFrame(rows).to_excel(path, sheet_name="Schedule", index=False)

    # ---------------------------------------------------------
    # 2️⃣ Resource Utilization (unchanged)
    # ---------------------------------------------------------
    def export_resource_utilization(self, output_dir: str, freq: str = 'D'):
        os.makedirs(output_dir, exist_ok=True)

        worker_rows, equipment_rows = [], []

        # Worker allocations
        for res_name, allocations in self.worker_manager.allocations.items():
            for (task_id, resource_name, units_used, start, end) in allocations:
                current_date = pd.to_datetime(start).normalize()
                while current_date <= pd.to_datetime(end).normalize() - pd.Timedelta(days=1):
                    worker_rows.append({
                        "Date": current_date,
                        "TaskID": task_id,
                        "Resource": resource_name,
                        "UnitsUsed": units_used
                    })
                    current_date += pd.Timedelta(days=1)

        # Equipment allocations
        for res_name, allocations in self.equipment_manager.allocations.items():
            for (task_id, resource_name, units_used, start, end) in allocations:
                current_date = pd.to_datetime(start).normalize()
                while current_date <= pd.to_datetime(end).normalize() - pd.Timedelta(days=1):
                    equipment_rows.append({
                        "Date": current_date,
                        "TaskID": task_id,
                        "Resource": resource_name,
                        "UnitsUsed": units_used
                    })
                    current_date += pd.Timedelta(days=1)

        def _process_and_export(rows, filename):
            df = pd.DataFrame(rows)
            if df.empty:
                df = pd.DataFrame(columns=["Date", "TaskID", "Resource", "UnitsUsed"])
            else:
                df['Date'] = pd.to_datetime(df['Date'])
                if freq == 'W':
                    df = df.groupby(
                        [pd.Grouper(key='Date', freq='W-MON'), 'TaskID', 'Resource'],
                        as_index=False
                    )['UnitsUsed'].sum()
                else:
                    df = df.sort_values(['Date', 'Resource', 'TaskID'])
                    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')

            output_path = os.path.join(output_dir, filename)
            df.to_excel(output_path, sheet_name="ResourceUtilization", index=False)
            logger.info(f"✅ Resource utilization file exported: {output_path}")

        _process_and_export(worker_rows, "timephased_worker_usage.xlsx")
        _process_and_export(equipment_rows, "timephased_equipment_usage.xlsx")

    # ---------------------------------------------------------
    # 3️⃣ CPM Export (unchanged)
    # ---------------------------------------------------------
    def export_cpm(self, path):
        from logic import CPMAnalyzer
        durations = {t.id: max(1, (self.schedule[t.id][1] - self.schedule[t.id][0]).days)
                     for t in self.tasks if t.id in self.schedule}
        dependencies = {t.id: t.predecessors for t in self.tasks}
        cpm = CPMAnalyzer(list(durations.keys()), durations, dependencies).run()
     
        rows = []
        for t in self.tasks:
            if t.id in cpm.ES:
                es_days, ef_days = cpm.ES[t.id], cpm.EF[t.id]
                ls_days, lf_days = cpm.LS[t.id], cpm.LF[t.id]

                es_date = self.calendar.add_workdays(self.calendar.current_date, es_days)
                ef_date = pd.to_datetime(self.calendar.add_workdays(self.calendar.current_date, ef_days)) - pd.Timedelta(days=1)
                ls_date = self.calendar.add_workdays(self.calendar.current_date, ls_days)
                lf_date = pd.to_datetime(self.calendar.add_workdays(self.calendar.current_date, lf_days)) - pd.Timedelta(days=1)

                rows.append({
                    "TaskID": t.id,
                    "TaskName": t.name,
                    "Discipline": t.discipline,
                    "DurationDays": durations[t.id],
                    "ES_days": es_days, "ES_date": pd.to_datetime(es_date).date(),
                    "EF_days": ef_days, "EF_date": pd.to_datetime(ef_date).date(),
                    "LS_days": ls_days, "LS_date": pd.to_datetime(ls_date).date(),
                    "LF_days": lf_days, "LF_date": pd.to_datetime(lf_date).date(),
                    "Float": cpm.float[t.id],
                    "Critical": "Yes" if cpm.float[t.id] == 0 else "No",
                    "ScheduledStart": pd.to_datetime(self.schedule[t.id][0]).date(),
                    "ScheduledEnd": (pd.to_datetime(self.schedule[t.id][1]) - pd.Timedelta(days=1)).date(),
                })
        pd.DataFrame(rows).to_excel(path, sheet_name="CPM", index=False)

    # ---------------------------------------------------------
    # 4️⃣ New: Weekly Task Progress (overlap-based)
    # ---------------------------------------------------------
    def export_weekly_task_progress(self, path):
        rows = []

        for t in self.tasks:
            if t.id not in self.schedule:
                continue

            s = pd.to_datetime(self.schedule[t.id][0])
            e = pd.to_datetime(self.schedule[t.id][1])
            duration = max(1, (e - s).days + 1)
            cost = getattr(t, "cost", 1.0)
            crit_weight = 2.0 if getattr(t, "is_critical", False) else 1.0
            
            # Generate all week boundaries (Mondays)
            #weeks = pd.date_range(s.floor("W-MON"), e.ceil("W-MON"), freq="W-MON")
            week_start = s - pd.Timedelta(days=s.weekday())
            week_end = e + pd.Timedelta(days=(6 - e.weekday()))
            weeks = pd.date_range(week_start, week_end, freq="W-MON")
            cumulative = 0.0
            for week_start in weeks:
                week_end = week_start + pd.Timedelta(days=6)

                overlap_start = max(s, week_start)
                overlap_end = min(e, week_end)
                overlap_days = max(0, (overlap_end - overlap_start).days + 1)

                weekly_progress = overlap_days / duration
                cumulative += weekly_progress

                rows.append({
                    "TaskID": t.id,
                    "TaskName": t.name,
                    "Discipline": t.discipline,
                    "Zone": t.zone,
                    "Floor": t.floor,
                    "WeekStart": week_start.date(),
                    "WeekEnd": week_end.date(),
                    "WeeklyProgress": round(weekly_progress, 4),
                    "CumulativeProgress": round(min(cumulative, 1.0), 4),
                    "Duration_days": duration,
                    "Cost": cost,
                    "CritWeight": crit_weight
                })

        pd.DataFrame(rows).to_excel(path, sheet_name="WeeklyProgress", index=False)
        logger.info(f"✅ Weekly task progress exported: {path}")

    # ---------------------------------------------------------
    # 5️⃣ New: Weekly Discipline Progress (aggregated)
    # ---------------------------------------------------------
    def export_weekly_discipline_progress(self, task_progress_path, output_path):
        df = pd.read_excel(task_progress_path)

        # Weighted progress aggregation
        df["DurationWeighted"] = df["WeeklyProgress"] * df["Duration_days"]
        df["CostWeighted"] = df["WeeklyProgress"] * df["Cost"]
        df["CritInfluence"] = df["WeeklyProgress"] * df["CritWeight"]

        agg = df.groupby(
            ["Discipline", "Zone", "Floor", "WeekStart"], as_index=False
        ).agg({
            "DurationWeighted": "sum",
            "CostWeighted": "sum",
            "CritInfluence": "sum"
        })

        # Normalize each weighting separately to 0–1
        agg["DurationWeighted"] /= agg.groupby(["Discipline", "Zone", "Floor"])["DurationWeighted"].transform("sum")
        agg["CostWeighted"] /= agg.groupby(["Discipline", "Zone", "Floor"])["CostWeighted"].transform("sum")
        agg["CritInfluence"] /= agg.groupby(["Discipline", "Zone", "Floor"])["CritInfluence"].transform("sum")

        # Compute cumulative values
        agg["CumDurationProgress"] = agg.groupby(["Discipline", "Zone", "Floor"])["DurationWeighted"].cumsum()
        agg["CumCostProgress"] = agg.groupby(["Discipline", "Zone", "Floor"])["CostWeighted"].cumsum()
        agg["CumCriticalityProgress"] = agg.groupby(["Discipline", "Zone", "Floor"])["CritInfluence"].cumsum()

        agg.to_excel(output_path, sheet_name="DisciplineWeeklyProgress", index=False)
        logger.info(f"✅ Weekly discipline progress exported: {output_path}")

    # ---------------------------------------------------------
    # 6️⃣ Export all reports together
    # ---------------------------------------------------------
    def export_all(self, folder=None):
        if folder is None:
            folder = tempfile.mkdtemp(prefix="schedule_output_")
        os.makedirs(folder, exist_ok=True)

        # Schedule + Resource + CPM
        self.export_schedule(os.path.join(folder, "construction_schedule_optimized.xlsx"))
        self.export_resource_utilization(folder)
        self.export_cpm(os.path.join(folder, "critical_path_cpm.xlsx"))

        # Weekly progress reports
        weekly_task_path = os.path.join(folder, "weekly_task_progress.xlsx")
        weekly_disc_path = os.path.join(folder, "weekly_discipline_progress.xlsx")

        self.export_weekly_task_progress(weekly_task_path)
        self.export_weekly_discipline_progress(weekly_task_path, weekly_disc_path)
        try:
            gantt_html_path = os.path.join(folder, "interactive_gantt.html")
            generate_interactive_gantt(self.schedule, gantt_html_path)
            print(f"🗂️ Interactive Gantt saved: {gantt_html_path}")
        except Exception as e:
            print(f"⚠️ Failed to generate Gantt: {e}")

        print(f"✅ All reports exported to folder: {folder}")
        return folder

import plotly.graph_objects as go
import plotly.express as px

class MonitoringReporter:
    def __init__(self, reference_schedule: pd.DataFrame, actual_progress: pd.DataFrame):
        """
        reference_schedule: DataFrame with columns [TaskID, TaskName, Start, End]
        actual_progress: DataFrame with columns [Date, TaskID, Progress] (0-1)
        """
        self.ref_df = reference_schedule.copy()
        self.act_df = actual_progress.copy()
        self.analysis_df = None

        # Ensure proper datetime
        self.ref_df["Start"] = pd.to_datetime(self.ref_df["Start"])
        self.ref_df["End"] = pd.to_datetime(self.ref_df["End"])
        self.act_df["Date"] = pd.to_datetime(self.act_df["Date"])

    def compute_analysis(self):
        """Computes cumulative planned vs actual and deviation"""
        timeline = pd.date_range(self.ref_df["Start"].min(), self.ref_df["End"].max(), freq="D")
        planned_curve = []

        for day in timeline:
            ongoing = self.ref_df[(self.ref_df["Start"] <= day) & (self.ref_df["End"] >= day)]
            progress = len(ongoing) / len(self.ref_df)
            planned_curve.append({"Date": day, "PlannedProgress": progress})

        planned_df = pd.DataFrame(planned_curve)

        actual_df = self.act_df.groupby("Date", as_index=False)["Progress"].mean()
        actual_df["CumulativeActual"] = actual_df["Progress"].cumsum()
        actual_df["CumulativeActual"] = actual_df["CumulativeActual"].clip(upper=1.0)

        self.analysis_df = pd.merge(planned_df, actual_df, on="Date", how="outer").fillna(method="ffill")
        self.analysis_df["ProgressDeviation"] = self.analysis_df["CumulativeActual"] - self.analysis_df["PlannedProgress"]

        return self.analysis_df

    def generate_scurve(self):
        if self.analysis_df is None:
            self.compute_analysis()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.analysis_df["Date"], y=self.analysis_df["PlannedProgress"],
            mode='lines', name='Planned', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=self.analysis_df["Date"], y=self.analysis_df["CumulativeActual"],
            mode='lines+markers', name='Actual', line=dict(color='green')
        ))
        fig.update_layout(
            title="S-Curve: Planned vs Actual Progress",
            xaxis_title="Date", yaxis_title="Cumulative Progress",
            template="plotly_white"
        )
        return fig

    def generate_deviation_chart(self):
        if self.analysis_df is None:
            self.compute_analysis()

        fig = px.bar(
            self.analysis_df, x="Date", y="ProgressDeviation",
            title="Progress Deviation (Actual - Planned)",
            labels={"ProgressDeviation": "Deviation"}
        )
        fig.update_layout(template="plotly_white")
        return fig

    def export_analysis_csv(self, file_path: str):
        if self.analysis_df is None:
            self.compute_analysis()
        self.analysis_df.to_csv(file_path, index=False)