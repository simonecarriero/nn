use std::cell::RefCell;
use std::collections::HashSet;
use std::f64::consts::E;
use std::rc::Rc;
use uuid::Uuid;

#[derive(Debug, PartialEq)]
pub struct Value {
    pub data: RefCell<f64>,
    label: Uuid,
    grad: RefCell<f64>,
    back: Op,
}

#[derive(Debug, PartialEq)]
enum Op {
    None,
    Sum { x: Rc<Value>, y: Rc<Value> },
    Mul { x: Rc<Value>, y: Rc<Value> },
    Tanh { x: Rc<Value> },
}

pub fn value(data: f64) -> Rc<Value> {
    value_with_grad(data, 0.0)
}

fn value_with_grad(data: f64, grad: f64) -> Rc<Value> {
    Rc::new(Value { label: Uuid::new_v4(), data: RefCell::new(data), grad: RefCell::new(grad), back: Op::None })
}

pub fn add(x: &Rc<Value>, y: &Rc<Value>) -> Rc<Value> {
    Rc::new(Value {
        label: Uuid::new_v4(),
        data: RefCell::new(*x.data.borrow() + *y.data.borrow()),
        grad: RefCell::new(0.0),
        back: Op::Sum { x: Rc::clone(x), y: Rc::clone(y) },
    })
}

pub fn mul(x: &Rc<Value>, y: &Rc<Value>) -> Rc<Value> {
    Rc::new(Value {
        label: Uuid::new_v4(),
        data: RefCell::new(*x.data.borrow() * *y.data.borrow()),
        grad: RefCell::new(0.0),
        back: Op::Mul { x: Rc::clone(x), y: Rc::clone(y) },
    })
}

pub fn tanh(x: &Rc<Value>) -> Rc<Value> {
    let d = *x.data.borrow();
    Rc::new(Value {
        label: Uuid::new_v4(),
        data: RefCell::new((E.powf(2.0 * d) - 1.0) / (E.powf(2.0 * d) + 1.0)),
        grad: RefCell::new(0.0),
        back: Op::Tanh { x: Rc::clone(x) },
    })
}

fn prev(value: &Rc<Value>) -> Vec<&Rc<Value>> {
   match &value.back {
       Op::None => vec![],
       Op::Sum { x, y } => vec![x, y],
       Op::Mul { x, y } => vec![x, y],
       Op::Tanh { x } => vec![x],
   }
}

fn topological_sort<'a>(value: &'a Rc<Value>, topo: &mut Vec<&'a Rc<Value>>, visited: &mut HashSet<Uuid>) {
    if !visited.contains(&value.label) {
        visited.insert(value.label);
        for v in prev(value) {
            topological_sort(v, topo, visited)
        }
        topo.push(value)
    }
}

fn backward(value: &Rc<Value>) {
    *value.grad.borrow_mut() = 1.0;
    let mut topo = vec![];
    topological_sort(value, &mut topo, &mut HashSet::new());

    for v in topo.iter().rev() {
        backward_step(v);
    }
}

fn backward_step(value: &Rc<Value>) {
    match &value.back {
        Op::None => {}
        Op::Sum { x, y } => {
            let x_local_derivative = 1.0;
            let y_local_derivative = 1.0;
            *x.grad.borrow_mut() += x_local_derivative * *value.grad.borrow();
            *y.grad.borrow_mut() += y_local_derivative * *value.grad.borrow();
        }
        Op::Mul { x, y } => {
            let x_local_derivative = *y.data.borrow();
            let y_local_derivative = *x.data.borrow();
            *x.grad.borrow_mut() += x_local_derivative * *value.grad.borrow();
            *y.grad.borrow_mut() += y_local_derivative * *value.grad.borrow();
        }
        Op::Tanh { x } => {
            let x_local_derivative = 1.0 -  &value.data.borrow().powi(2);
            *x.grad.borrow_mut() += x_local_derivative * *value.grad.borrow();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expression_builds_dag() {
        let a = value(0.1);
        let b = value(0.2);
        let c = value(0.3);
        let x = tanh(&add(&a, &mul(&b, &c)));

        assert_eq!(*x.data.borrow(), 0.15864850429749888);
        if let Op::Tanh { x } = &x.back {
            assert_eq!(*x.data.borrow(), 0.16);
            if let Op::Sum { x, y } = &x.back {
                assert_eq!(*x.data.borrow(), 0.1);
                assert_eq!(x.back, Op::None);
                assert_eq!(*y.data.borrow(), 0.06);
                if let Op::Mul { x, y} = &y.back {
                    assert_eq!(*x.data.borrow(), 0.2);
                    assert_eq!(*y.data.borrow(), 0.3);
                } else {
                    assert!(false, "unexpected Op")
                }
            } else {
                assert!(false, "unexpected Op")
            }
        } else {
            assert!(false, "unexpected Op")
        }
    }

    #[test]
    fn backward_step_on_sum_node_should_add_gradient_to_the_variables() {
        let a = value_with_grad(0.1, 0.08);
        let b = value_with_grad(0.2, 0.58);
        let x = add(&a, &b);
        *x.grad.borrow_mut() = 0.42;

        backward_step(&x);

        assert_eq!(*a.grad.borrow(), 0.5);
        assert_eq!(*b.grad.borrow(), 1.0);
    }

    #[test]
    fn backward_step_on_mul_node_should_add_gradient_to_the_variables() {
        let a = value_with_grad(0.5, 0.05);
        let b = value_with_grad(0.25, 0.05);
        let x = mul(&a, &b);
        *x.grad.borrow_mut() = 0.8;

        backward_step(&x);

        assert_eq!(*a.grad.borrow(), 0.25);
        assert_eq!(*b.grad.borrow(), 0.45);
    }

    #[test]
    fn backward_step_on_tanh_node_should_add_gradient_to_the_variable() {
        let a = value(0.8814);
        let x = tanh(&a);
        *x.grad.borrow_mut() = 1.0;

        backward_step(&x);

        assert_eq!(*a.grad.borrow(), 0.4999813233768232);
    }

    #[test]
    fn backward_on_dag_should_execute_on_topological_sort() {
        let a = value(3.0);
        let b = value(3.0);
        let c = value(2.0);
        let x = add(&mul(&a, &b), &mul(&c, &mul(&a, &b)));

        backward(&x);

        assert_eq!(*a.grad.borrow(), 9.0);
        assert_eq!(*b.grad.borrow(), 9.0);
        assert_eq!(*c.grad.borrow(), 9.0);
    }
}
