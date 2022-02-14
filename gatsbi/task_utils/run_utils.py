def _update_defaults(defaults_dict, unknown_list):
    print(unknown_list)
    keys_vals = [s.split("=") for s in unknown_list]
    print(keys_vals[0])
    unknown_keys = [s[0][2:] for s in keys_vals]
    unknown_vals = [s[1] for s in keys_vals]

    for key, val in zip(unknown_keys, unknown_vals):
        assert key in defaults_dict.keys()

        if type(defaults_dict[key]) == list:
            strip_val = val.strip(",").strip("[").strip("]")
            defaults_dict[key] = [int(a) for a in strip_val.split(",")]
        elif type(defaults_dict[key]) == dict:
            # split str to get key and value for dict entry
            strip_val = val.strip(" ").strip("{").strip("}")
            # split to get diff dict entries
            dict_list = [a.split(":") for a in strip_val.split(",")]
            defaults_dict[key] = {
                a[0].strip(" "): float(a[1]) for a in dict_list
            }  # reconstruct dict
        else:
            defaults_dict[key] = type(defaults_dict[key])(val)

    return defaults_dict
