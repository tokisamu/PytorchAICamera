
# Define training Pipeline
def train_model(model, criterion, optimizer, scheduler, num_epochs=1):    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))       
        # Each epoch has a training and validation phase
        for phase in ['train', 'varify']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # 训练模式
            else:
                model.train(False)  # 验证模式
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            iter=0
            for data in image_loader[phase]:
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                for i in preds:
                    print(class_names[labels[i]]+' '+class_names[preds[i]])
                loss = criterion(outputs, labels)
                print("phase:%s, epoch:%d/%d  Iter %d: loss=%s"%(phase,epoch,num_epochs-1,iter,str(loss.data.numpy())))
                # backward
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)
                iter += 1
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print('-' * 10)       
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model