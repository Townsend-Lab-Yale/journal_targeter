const metric_name = params.metric;
const show_all_metrics = params.show_all_metrics;

for (const metric in metric_col_inds) {
    const col = table_cols[metric_col_inds[metric]];
    if (show_all_metrics){
        col.visible = true;
    } else {
        col.visible = metric === metric_name;
    }
}
