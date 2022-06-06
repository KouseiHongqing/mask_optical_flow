def get_model(DEVICE,args):
    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model,map_location=DEVICE).state_dict())
    print(f"Loaded checkpoint at {args.model}")
    model = model.module
    model.to(DEVICE)
    model.eval()
    return model