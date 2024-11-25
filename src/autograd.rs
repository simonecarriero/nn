use std::cell::RefCell;
use std::rc::Rc;


#[derive(Debug, PartialEq)]
struct Value {
    data: RefCell<f64>,
    grad: RefCell<f64>,
    back: Op,
}

#[derive(Debug, PartialEq)]
enum Op {
    None,
    Sum { x: Rc<Value>, y: Rc<Value> },
    Mul { x: Rc<Value>, y: Rc<Value> }
}

fn value(data: f64) -> Rc<Value> {
    Rc::new(Value { data: RefCell::new(data), grad: RefCell::new(0.0), back: Op::None })
}

fn value_with_grad(data: f64, grad: f64) -> Rc<Value> {
    Rc::new(Value { data: RefCell::new(data), grad: RefCell::new(grad), back: Op::None })
}

fn add(x: &Rc<Value>, y: &Rc<Value>) -> Rc<Value> {
    Rc::new(Value {
        data: RefCell::new(*x.data.borrow() + *y.data.borrow()),
        grad: RefCell::new(0.0),
        back: Op::Sum { x: Rc::clone(x), y: Rc::clone(y) },
    })
}

fn mul(x: &Rc<Value>, y: &Rc<Value>) -> Rc<Value> {
    Rc::new(Value {
        data: RefCell::new(*x.data.borrow() * *y.data.borrow()),
        grad: RefCell::new(0.0),
        back: Op::Mul { x: Rc::clone(x), y: Rc::clone(y) },
    })
}

fn backward(value: &Rc<Value>) {
    match &value.back {
        Op::None => {}
        Op::Sum { x, y } => {
            let x_local_derivative = 1.0;
            let y_local_derivative = 1.0;
            *x.grad.borrow_mut() += x_local_derivative * *value.grad.borrow();
            *y.grad.borrow_mut() += y_local_derivative * *value.grad.borrow();
            backward(x);
            backward(y);
        }
        Op::Mul { x, y } => {
            let x_local_derivative = *y.data.borrow();
            let y_local_derivative = *x.data.borrow();
            *x.grad.borrow_mut() += x_local_derivative * *value.grad.borrow();
            *y.grad.borrow_mut() += y_local_derivative * *value.grad.borrow();
            backward(x);
            backward(y);
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
        let x = add(&a, &mul(&b, &c));

        assert_eq!(x, Rc::new(Value {
            data: RefCell::new(0.16),
            grad: RefCell::new(0.0),
            back: Op::Sum {
                x: value(0.1),
                y: Rc::new(Value {
                    data: RefCell::new(0.06),
                    grad: RefCell::new(0.0),
                    back: Op::Mul {
                        x: value(0.2),
                        y: value(0.3),
                    },
                }),
            },
        }));
    }

    #[test]
    fn backward_on_sum_node_should_add_gradient_to_the_variables() {
        let a = value_with_grad(0.1, 0.08);
        let b = value_with_grad(0.2, 0.58);
        let x = add(&a, &b);
        *x.grad.borrow_mut() = 0.42;

        backward(&x);

        assert_eq!(*a.grad.borrow(), 0.5);
        assert_eq!(*b.grad.borrow(), 1.0);
    }

    #[test]
    fn backward_on_mul_node_should_add_gradient_to_the_variables() {
        let a = value_with_grad(0.5, 0.05);
        let b = value_with_grad(0.25, 0.05);
        let x = mul(&a, &b);
        *x.grad.borrow_mut() = 0.8;

        backward(&x);

        assert_eq!(*a.grad.borrow(), 0.25);
        assert_eq!(*b.grad.borrow(), 0.45);
    }

    #[test]
    fn backward_on_add_node_should_propagate_to_downstream_dag() {
        let a = value(0.0);
        let b = value(0.0);
        let x = add(&add(&a, &b), &add(&a, &b));
        *x.grad.borrow_mut() = 1.0;

        backward(&x);

        assert_eq!(*a.grad.borrow(), 2.0);
        assert_eq!(*b.grad.borrow(), 2.0);
    }

    #[test]
    fn backward_on_mul_node_should_propagate_to_downstream_dag() {
        let a = value(0.2);
        let b = value(0.3);
        let x = mul(&add(&a, &b), &add(&a, &b));
        *x.grad.borrow_mut() = 1.0;

        backward(&x);

        assert_eq!(*a.grad.borrow(), 1.0);
        assert_eq!(*b.grad.borrow(), 1.0);
    }
}
