import cupy as cp

def test(net, x_test, t_test, batch_size=2048):
    net.eval()
    
    images = cp.array(x_test)
    labels = cp.array(t_test)
    
    total_samples = images.shape[0]
    num_batches = (total_samples + batch_size - 1) // batch_size
    correct_count = 0
    
    with cp.cuda.Device(0): 
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            
            batch_images = images[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            
            infers = net.inference(batch_images.reshape(-1, 1, 28, 28))
            
            gt_list = cp.argmax(batch_labels, axis=1)
            correct_count += cp.sum(infers == gt_list)
    
    accuracy = cp.asnumpy(correct_count / float(total_samples))
    
    return accuracy