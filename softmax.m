function y = softmax(x)
    exp_x = exp(x-max(x));
    y = exp_x / sum(exp_x);
end



