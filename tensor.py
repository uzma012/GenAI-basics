import torch

x = torch.linspace(0, 10, 100).reshape(-1,1 )
y = 3*x+2+torch.randn_like(x)*0.5

W = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
lr = 0.01
epochs = 200

for epoch in range(epochs):
    y_pred = W*x + b
    loss = torch.mean((y - y_pred)**2)
    loss.backward()
    
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
    W.grad.zero_()
    b.grad.zero_()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, W: {W.item():.4f}, b: {b.item():.4f}')


print(f'Final parameters: W: {W.item():.4f}, b: {b.item():.4f}')