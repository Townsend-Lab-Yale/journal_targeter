const metric_name = params.metric;
const weight = params.weight;
const impact_vals = source.data[metric_name];
const cat_vals = source.data['CAT'];
const show_all_metrics = params.show_all_metrics;

const new_data = Object.assign({}, source.data);

let max_impact = 0;
if (changed_metric){
    let na_vals = [];
    for (let i = 0; i < impact_vals.length; i++) {
        let impact = impact_vals[i];
        if (impact > max_impact){
            max_impact = impact;
        }
        if (impact < 0){
            na_vals.push('whitesmoke');
        }
        else {
            na_vals.push('white');
        }
    }
    new_data.ax_impact = impact_vals;
    new_data.impact_max = Array(impact_vals.length).fill(max_impact);
    new_data.ax_impact_bg = na_vals;
    new_data.label_metric = source.data['label_'.concat(metric_name)];
    new_data.dominant = source.data['dominant_'.concat(metric_name)];
}

/* SET PROSPECT WHEN WEIGHT OR IMPACT CHANGES */
let prospects = [];
let expect = [];
for (let i = 0; i < impact_vals.length; i++) {
    let impact = impact_vals[i];
    if (impact >= 0){
        let cat = cat_vals[i];
        let p = cat / (weight * impact + cat);
        let pi = p * impact;
        prospects.push(p);
        expect.push(pi);
    }
    else {
        prospects.push(-1);
        expect.push(-1);
    }
}
new_data.prospect = prospects;
new_data.expect = expect;

/* UPDATE DATA SOURCE */
source.data = new_data;

if(changed_metric){
    for (let i = 0; i < metric_axes.length; i++){
        let ax = metric_axes[i];
        ax[0].axis_label = metric_name;
    }
    xrange.start = max_impact; /* XRANGE in ICATS-left */
    p.x_range.reset(); /* PROSPECT AXES */
    p.y_range.reset();

    if (! show_all_metrics) {
        /* Hide non-preferred columns */
        for (const metric in metric_col_inds) {
            const col = table_cols[metric_col_inds[metric]];
            col.visible = metric === metric_name;
        }
    }
}
