``Error with the upload api. `UploadFile` is not JSON serializable.``

This is due to the design choice, we need to switch back this endpoint to not use celery.

