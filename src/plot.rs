use plotly::layout::{Annotation, Axis, HAlign};
use plotly::{Contour, ImageFormat, Layout, Plot, Scatter};
use plotly::color::NamedColor;
use plotly::common::{ColorScale, ColorScaleElement, Line, Marker, Mode};
use crate::nn::Mlp;

pub fn plot_decision_boundary(model: &Mlp, scores: &[(f64, f64, f64)], text_top: &str, text_bottom: &str, out: &str) {
    let contour = contour(model, scores);
    let (blue, red) = scatter(scores);
    let mut plot = Plot::new();
    plot.add_trace(contour);
    plot.add_trace(blue);
    plot.add_trace(red);
    plot.set_layout(layout(vec![
        (0.05, 1.075, text_top),
        (0.05, -0.15, text_bottom)
    ]));
    plot.write_image(out, ImageFormat::PNG, 800, 600, 1.0);
}

pub fn plot_classification(scores: &[(f64, f64, f64)], text_bottom: &str, out: &str) {
    let (blue, red) = scatter(scores);
    let mut plot = Plot::new();
    plot.add_trace(blue);
    plot.add_trace(red);
    plot.set_layout(layout(vec![(0.0, 0.0, text_bottom)]));
    plot.write_image(out, ImageFormat::PNG, 800, 600, 1.0);
}

fn contour(model: &Mlp, scores: &[(f64, f64, f64)]) -> Box<Contour<f64>> {
    let all_x = scores.iter().map(|(x, _, _)| *x).collect::<Vec<_>>();
    let max_x = all_x.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let min_x = all_x.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let x_step = (max_x - min_x) / 20.0;

    let all_y = scores.iter().map(|(_, y, _)| *y).collect::<Vec<_>>();
    let max_y = all_y.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    let min_y = all_y.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    let y_step = (max_y - min_y) / 20.0;

    let (mut grid_xs, mut grid_ys, mut grid_zs) = (vec![], vec![], vec![]);
    let mut x = *min_x;
    while x < *max_x + 0.001 {
        let mut y = *min_y;
        while y < *max_y + 0.001 {
            let score = model.process(&[x, y]);
            grid_xs.push(x);
            grid_ys.push(y);
            grid_zs.push(*score[0].data.borrow());
            y += y_step;
        }
        x += x_step;
    }

    Contour::new(grid_xs, grid_ys, grid_zs)
        .n_contours(1)
        .show_legend(false)
        .show_scale(false)
        .line(Line::new().width(0.0).smoothing(1.0))
        .auto_color_scale(false)
        .color_scale(ColorScale::Vector(vec![
            ColorScaleElement(0., "#ff7675".to_string()),
            ColorScaleElement(1., "#74b9ff".to_string()),
        ]))
}

fn scatter(points: &[(f64, f64, f64)]) -> (Box<Scatter<f64, f64>>, Box<Scatter<f64, f64>>) {
    let (blue_points, red_points) = points.iter().partition::<Vec<(f64, f64, f64)>, _>(|(_, _, label)| *label > 0.0);
    let blue_xs = blue_points.iter().map(|(x, _, _)| *x).collect();
    let blue_ys = blue_points.iter().map(|(_, y, _)| *y).collect();
    let red_xs = red_points.iter().map(|(x, _, _)| *x).collect();
    let red_ys = red_points.iter().map(|(_, y, _)| *y).collect();
    let blue = Scatter::new(blue_xs, blue_ys).mode(Mode::Markers).marker(Marker::new().size(5).color(NamedColor::Blue));
    let red = Scatter::new(red_xs, red_ys).mode(Mode::Markers).marker(Marker::new().size(5).color(NamedColor::Red));
    (blue, red)
}

pub fn layout(annotations: Vec<(f64, f64, &str)>) -> Layout {
    Layout::new()
        .x_axis(Axis::new().visible(false).show_grid(false))
        .y_axis(Axis::new().visible(false).show_grid(false))
        .annotations(annotations.iter().map(|(x, y, z)|
            Annotation::new()
                .text(*z)
                .x_ref("paper").x(*x)
                .y_ref("paper").y(*y)
                .align(HAlign::Left)
                .font(plotly::common::Font::new().size(20).family("Courier New"))
                .show_arrow(false)
        ).collect())
        .show_legend(false)
}
