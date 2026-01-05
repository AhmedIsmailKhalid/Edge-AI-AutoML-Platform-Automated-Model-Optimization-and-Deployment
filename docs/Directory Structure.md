# Directory Structure

```
edge-ai-automl-platform/
├── .coverage
├── .env
├── .env.example
├── .gitignore
├── package.json
├── package-lock.json
├── poetry.lock
├── pyproject.toml
├── README.md
├── Version History.md
├── dataset
│   ├── custom
│   │   ├── .gitkeep
│   │   ├── pytorch
│   │   └── tensorflow
│   └── preset
│       ├── .gitkeep
│       ├── cifar-100-python.tar.gz
│       ├── cifar-10-python.tar.gz
│       ├── cifar10
│       ├── cifar-100-python
│       ├── cifar-10-batches-py
│       ├── FashionMNIST
│       └── MNIST
├── docs
│   ├── Directory Structure.md
│   ├── Implementation Workflow.md
│   ├── System Architecture.md
│   └── Tech Stack.md
├── frontend
├── models
│   ├── custom
│   │   ├── .gitkeep
│   │   ├── pytorch
│   │   └── tensorflow
│   ├── optimized
│   │   ├── .gitkeep
│   │   ├── 03f8915d-5e83-4299-a531-bea5fc8479da
│   │   ├── 041987c7-c68e-4eb5-afdd-3741346a0a9d
│   │   ├── 06634130-3fec-4bb4-a092-7c311a608001
│   │   ├── 070d0243-084b-498d-a5f8-9829ae7b2265
│   │   ├── 0c5503c7-a7b5-428b-a846-4fe8a8857cfb
│   │   ├── 0de4b03e-f28c-4502-8a8a-cdb75b8149c5
│   │   ├── 0e7c9c48-f394-4739-a136-23e8400cd206
│   │   ├── 13a1a149-63ce-4991-b2c1-e095a5f1f21e
│   │   ├── 13a92ea6-8035-4e51-a373-7c22246fea11
│   │   ├── 17cca18f-331d-4634-9148-2b6bbac77d17
│   │   ├── 1d0c0e08-4b2d-4363-b221-6cf5ff3fa4b4
│   │   ├── 1f698f92-c37b-4698-9996-db2e120b7593
│   │   ├── 22ebf05d-2023-4cc8-96a7-d1b282c445a1
│   │   ├── 26cd9d44-5c08-4368-860b-9283b1ef0cc6
│   │   ├── 27a1802e-33d7-4b05-86eb-7318508d10b0
│   │   ├── 2aebb1ea-349f-468e-989b-28e4a135f9d0
│   │   ├── 2c28cd73-14b1-4fef-ae50-4e8881ad0f53
│   │   ├── 2d2952da-0365-4999-8173-9b29379a0b33
│   │   ├── 37bcf620-002f-4eb9-a5a0-4977b43c8e74
│   │   ├── 384473c5-4d88-4d17-a87d-71eaa5e9b07f
│   │   ├── 39fba0a8-cf3a-4375-aa0d-917078a0d486
│   │   ├── 3b7e2b79-abad-48ae-ac6a-32dc4e849e36
│   │   ├── 405072d1-2f6b-4ba8-a9dc-38119fc358ec
│   │   ├── 4355e135-41bf-4ebd-8b31-afe37df83387
│   │   ├── 43bfe933-cdd2-48fc-9526-0fe4d7b1d06f
│   │   ├── 4499d7c8-99bc-4396-ac6f-fb41aee18f02
│   │   ├── 4565d3fa-9c54-4c9c-b176-9e3185109251
│   │   ├── 460dacf7-4e2e-4551-8af5-d8179faca48b
│   │   ├── 466cc014-30ee-4eff-a5f2-b807bd9c97d0
│   │   ├── 495cf443-af77-475e-a1c6-1b842b3e210d
│   │   ├── 49babd97-e49e-4b52-9082-853159973c74
│   │   ├── 4c62280f-3918-44fa-955d-b2d8b9cf159c
│   │   ├── 4fb58803-b75f-4ec4-9ff9-97a8bb1e4343
│   │   ├── 506b8c22-9899-4b66-8639-cf049000af41
│   │   ├── 50fab828-0d5e-4912-bac1-946cbeb23fdc
│   │   ├── 53497b29-3477-413a-900a-8b2b2bd44b67
│   │   ├── 544da7ce-a64e-45e7-9f47-cf0db25011e8
│   │   ├── 58351a4a-cbf2-4707-8142-e1a91561ae19
│   │   ├── 591bde98-8f6e-47ee-b783-398fc000938f
│   │   ├── 5b3411ca-89fd-4f46-ab9c-db2729d4b02f
│   │   ├── 5c905312-9632-4e54-bf33-1bd5ddc4d777
│   │   ├── 5e5ee1d1-9654-4544-9b12-f985800158d0
│   │   ├── 5f4430af-c6b7-41af-84e7-b99617bf96cf
│   │   ├── 5fdcb984-521d-4c65-974d-554d65a75fd5
│   │   ├── 6077ddb9-48b7-4208-81ec-0bbad3d28b0f
│   │   ├── 64638dba-2c25-4124-bbec-a42075650d7d
│   │   ├── 65041ae9-0e1d-4461-9bd5-cef92e3fce4b
│   │   ├── 66190378-8541-40cb-8adc-8d710f5ed39f
│   │   ├── 66fad5aa-91d6-4085-9366-49349e696a91
│   │   ├── 67d98971-b592-47bd-8764-6215f542b986
│   │   ├── 68e2d3c3-d3e1-4e08-b5fc-a8f9999126ed
│   │   ├── 6a17c676-71ab-4968-833c-64d07e417506
│   │   ├── 6af995be-7a54-4d18-adc2-4679746fdfc1
│   │   ├── 6b067e4e-d252-4083-8719-0bee1633f6d9
│   │   ├── 6c54cead-3d76-4ee6-a6b2-f1a845b3549d
│   │   ├── 6ce89e89-c374-4314-a4d1-37d54f843877
│   │   ├── 6cf44c59-6601-48bc-8cbe-059109dbf140
│   │   ├── 6dd26353-70c4-4cb9-a925-84e09a7de618
│   │   ├── 6ed6e571-3ff0-4d2f-9746-dc95167ceb3a
│   │   ├── 6f561931-787d-47f8-9b62-d4075699cfca
│   │   ├── 70c68b95-6955-443c-8758-30b72925d518
│   │   ├── 712637d5-7fe1-49b3-9b5c-8b73f4e602c7
│   │   ├── 732c91c1-59cc-446b-abc6-77c185849414
│   │   ├── 73992ff7-c88d-4cae-a8c0-952980e85b1e
│   │   ├── 75642a04-b865-4d64-93fe-1a2305024fb6
│   │   ├── 7bcd4449-8235-4e99-b431-926708e87500
│   │   ├── 7be1fc9d-424f-49a6-9475-76078070924b
│   │   ├── 7d18e69b-9e71-46d6-bd68-4c1d9c63eca7
│   │   ├── 7de034c9-e0dc-424f-a6c3-7c4342364dc5
│   │   ├── 7f839f22-97e5-4e7f-a4c3-38752a40a190
│   │   ├── 83c1159c-5b5a-4cde-b8f8-f9b13cbcb43c
│   │   ├── 87ac8a02-f483-4448-a9cf-c5b8a2acddad
│   │   ├── 888b123b-6792-432c-b844-4cc0a1d6d963
│   │   ├── 8ad5da9d-36c9-43aa-ad3c-dd2a822ffb57
│   │   ├── 8b070797-0056-43bf-a9b2-ed2afc650b36
│   │   ├── 8b928e13-85a8-46b5-831a-a2e43f9db4ac
│   │   ├── 8eedbd77-8420-440b-a97e-36686f0e11db
│   │   ├── 8f5a7f51-2c1f-4b12-9daa-49cd95f68fab
│   │   ├── 91ec6ad7-fa4d-45c8-bdee-eb02693875c6
│   │   ├── 93432ada-0a1a-46a3-855d-2d5ca1d2edfb
│   │   ├── 939655ec-452b-4a84-aa7a-bf1d02027789
│   │   ├── 961a7a43-d02d-41d7-9dcf-77383c3d2855
│   │   ├── 985a2903-f138-42d2-8c8d-f7694eed91f6
│   │   ├── 98f94629-3662-402b-ac77-fcce160294a0
│   │   ├── 9a194516-bf1e-4233-be5c-1e99e313fe91
│   │   ├── 9c18ba54-6194-4655-9eca-9566ed5e84d6
│   │   ├── a186fe15-2b31-4bba-95f8-286e9174be18
│   │   ├── a4995574-fb19-4d4a-98e9-c227400ba591
│   │   ├── a7cd5f89-2e7c-4a06-bb0d-380f3e89c6b8
│   │   ├── a81e3808-8576-4435-baf4-6deada273b18
│   │   ├── aa9b2f0c-6e14-4755-8ce8-6493fc9faf76
│   │   ├── ac5e1bb7-2fa0-4348-999a-0dd4312517fa
│   │   ├── ad120d47-83b7-46ff-ae1f-6450f05b7fb8
│   │   ├── ad8ad987-0a96-4c36-b3d2-afe44cd153d6
│   │   ├── ad9d0a12-1aab-4f45-907c-c28b8a90985e
│   │   ├── af0a982c-ea4b-4e7a-9b0b-d942f00173dd
│   │   ├── b084c616-b0c9-4ec6-8c8e-7702e76cbe2d
│   │   ├── b227f60c-7e26-433d-8195-e08ab4d478a2
│   │   ├── b2fa0984-6761-4e71-878f-0d2f055728ce
│   │   ├── b475323c-59d4-4888-9f64-12d32924ec3e
│   │   ├── b5a29306-b52a-4e05-80a8-a2bbed032e32
│   │   ├── b64222da-de9a-422e-b4fa-be5e4f6e8f00
│   │   ├── b8c02163-4552-4104-a805-b8caab3cf527
│   │   ├── b8d98080-1e7d-4e2d-92a9-d713d7a1bae7
│   │   ├── bc9bb19e-ad8d-4f02-8117-27d087ce75cb
│   │   ├── bcd1d262-1abd-4e1a-8846-30ecbd4e1392
│   │   ├── bd3d9b80-4f23-478c-8459-209dec77bbfc
│   │   ├── be017aa5-406d-448b-a5ac-0b9c3bab55e6
│   │   ├── c374a540-ee50-409d-be9f-d0d7fbe61ee0
│   │   ├── c60343b4-7bd3-47a0-89ee-e1e768e32a1b
│   │   ├── ca6be803-5639-4e03-8e54-504a546d535b
│   │   ├── cc981257-0a72-4f15-b7d0-df13d61d5154
│   │   ├── cefd6a44-93b6-4f7c-80b4-4addac00d397
│   │   ├── d145efb1-0885-4216-a3d6-a2603bf28ebb
│   │   ├── d219d61b-a9e0-43f9-b19f-b6ce940b3483
│   │   ├── d46dc7de-4043-4fca-80ed-80c6ccd70a85
│   │   ├── d5421fc8-8b33-4e4f-9153-75cc232d30df
│   │   ├── d7433f06-e0ff-4471-9dba-cc50aa682c04
│   │   ├── d7e709be-99b3-4191-bfc6-6dc866d20fb6
│   │   ├── dad86b3b-f5b7-4703-ad87-f918dcea622a
│   │   ├── db2324d3-c38d-433b-849b-dbb365e14d6f
│   │   ├── db5207b1-1866-4424-80d6-1c5666e199bd
│   │   ├── ddd8f8b1-6ffd-4cc8-849f-8e22bf7f5677
│   │   ├── df06f6b0-60e1-45f1-9519-84808c33d3ae
│   │   ├── df0f16da-e561-439d-9c77-77d5bc801c35
│   │   ├── e019d229-e71b-4fec-bd69-041dd8dea298
│   │   ├── e242641f-04c5-4716-aaf2-2834b0cb3f39
│   │   ├── e88ecdef-53e7-4f9c-8a5b-6f983df7697b
│   │   ├── e9a22415-a560-40f3-8b26-084aa8ba3f71
│   │   ├── ecfb80b8-7c34-42ce-a012-a664d5cdc345
│   │   ├── ee6ab90f-c40a-47b1-ba38-cbe528294317
│   │   ├── ef67e2fc-edd3-4c07-a01f-46617efa787f
│   │   ├── f062ea7a-4bf3-4c78-a4b0-2772bfa4f1d2
│   │   ├── f25b943c-dacf-48ef-9728-fc117c841d47
│   │   ├── f2a3b8ee-3157-4e6f-a0e5-a3b9fd483691
│   │   ├── f3d6ce57-31b3-4903-9734-cb613af5b09d
│   │   ├── f788b8a4-e217-48ec-91d3-1cad8fd3396a
│   │   ├── f7cfcdbd-a502-4080-aea4-2e741035cfb0
│   │   ├── fe9632f4-c416-4216-badb-b0a7c5145be2
│   │   ├── ff7f7450-aad4-4a5d-b9c9-dae82f6226ea
│   │   ├── ff97c6bb-d9c1-4c3a-ab8d-2d2cde51b6fe
│   │   └── ffb12262-ade9-4b22-b2b7-86501c9be5b1
│   └── pretrained
│       ├── .gitkeep
│       ├── pytorch
│       └── tensorflow
├── scripts
│   ├── automated_e2e_tests.py
│   ├── cli_interface.py
│   ├── create_pretrained_models.py
│   ├── quick_failed_tests.py
│   ├── seed_devices.py
│   ├── test_app_startup.py
│   ├── test_experiments_api.py
│   ├── test_failed_cases.py
│   └── test_upload_api.py
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   ├── main.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── experiments.py
│   │   ├── optimize.py
│   │   ├── performance.py
│   │   ├── results.py
│   │   ├── upload.py
│   │   └── websocket.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── orchestrator.py
│   │   ├── performance_estimator.py
│   │   ├── recommendation_engine.py
│   │   ├── pytorch
│   │   └── tensorflow
│   ├── models
│   │   ├── __init__.py
│   │   ├── experiment.py
│   │   ├── experiment_progress.py
│   │   ├── model_file.py
│   │   ├── optimization_run.py
│   │   └── recommendation.py
│   ├── schemas
│   │   ├── __init__.py
│   │   ├── experiment.py
│   │   ├── optimization.py
│   │   ├── recommendation.py
│   │   ├── result.py
│   │   ├── upload.py
│   │   └── websocket.py
│   └── utils
│       ├── __init__.py
│       ├── dataset_loader.py
│       ├── file_handler.py
│       ├── model_loader.py
│       └── validators.py
├── test_results
│   ├── failed_tests_20251125_232950.json
│   ├── failed_tests_20251126_002451.json
│   ├── quick_test_20251124_205212.json
│   ├── quick_test_20251124_212815.json
│   ├── quick_test_20251124_215719.json
│   ├── quick_test_20251125_111735.json
│   └── test_report_20251124_134915.txt
└── tests
    ├── test_models.py
    ├── e2e
    │   └── test_complete_workflow.py
    ├── integration
    │   ├── test_orchestrator.py
    │   └── test_performance_api.py
    └── unit
        ├── test_distillation_pytorch.py
        ├── test_distillation_tensorflow.py
        ├── test_hybrid_pytorch.py
        ├── test_hybrid_tensorflow.py
        ├── test_performance_estimator.py
        ├── test_pruning_pytorch.py
        ├── test_pruning_tensorflow.py
        ├── test_ptq_int4_pytorch.py
        ├── test_ptq_int4_tensorflow.py
        ├── test_ptq_int8_pytorch.py
        ├── test_ptq_int8_tensorflow.py
        ├── test_qat_pytorch.py
        ├── test_qat_tensorflow.py
        ├── test_recommendation_engine.py
        └── test_websocket_connection.py
```