import timm

avail_pretrained_models = timm.list_models("*resnet18*", pretrained=True)
print(len(avail_pretrained_models))

for i, model_name in enumerate(avail_pretrained_models):
    print(model_name, end=" ")

    if i % 5 == 0:
        print()
