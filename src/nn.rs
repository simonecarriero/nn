use std::rc::Rc;
use crate::autograd::{add, mul, tanh, value, Value};

struct Neuron {
    weights: Vec<Rc<Value>>,
    bias: Rc<Value>,
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Neuron {
    fn process(&self, inputs: &[&Rc<Value>]) -> Rc<Value> {
        let mut sum = value(0.0);
        for (wi, xi) in self.weights.iter().zip(inputs) {
            sum = add(&sum, &mul(wi, xi));
        }
        tanh(&add(&sum, &self.bias))
    }
}

impl Layer {
    fn process(&self, inputs: &[&Rc<Value>]) -> Vec<Rc<Value>> {
        self.neurons.iter().map(|n| n.process(&inputs)).collect()
    }
}

fn neuron(weights: &[f64], bias: f64) -> Neuron {
    Neuron { weights: weights.iter().map(|w| value(*w)).collect(), bias: value(bias) }
}

fn layer(number_of_neurons: i16, weights: &[f64], bias: f64) -> Layer {
    Layer { neurons: (0..number_of_neurons).map(|_| neuron(weights, bias)).collect()}
}

#[cfg(test)]
mod tests {
    use crate::autograd::{value};
    use crate::nn::*;

    #[test]
    fn neuron_should_process_input() {
        let weights = vec![0.4, 0.2, 0.7];
        let n = neuron(&weights, 0.2);
        let x1 = value(0.8);
        let x2 = value(0.7);
        let x3 = value(0.3);

        let output = n.process(&vec![&x1, &x2, &x3]);

        assert_eq!(*output.data.borrow(), 0.7013741309383126);
    }

    #[test]
    fn layer_should_process_input_forwarding_to_all_neurons() {
        let weights = vec![0.4, 0.2, 0.7];
        let l = layer(3, &weights, 0.2);
        let x1 = value(0.8);
        let x2 = value(0.7);
        let x3 = value(0.3);

        let output = l.process(&vec![&x1, &x2, &x3]);

        let outputs: Vec<_> = output.iter().map(|n| *n.data.borrow()).collect();
        assert_eq!(outputs, vec![
            0.7013741309383126,
            0.7013741309383126,
            0.7013741309383126,
        ]);
    }
}
