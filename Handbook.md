# TensorStore Python API Handbook (Detailed)

This document provides a comprehensive, detailed reference guide to the TensorStore Python API, with explanations and code examples for common usage patterns.

---

## **1. Top-Level Functions**

These are the main functions imported directly from the `tensorstore` module for creating and manipulating `TensorStore` objects.

### **`ts.open(spec, **kwargs)`**

The primary function for opening or creating a `TensorStore`. It operates asynchronously.

* **Description:** `ts.open` parses a specification object (`spec`) that describes the storage driver, location, and metadata of a multi-dimensional array. It returns a `Future` which, upon completion, yields a `TensorStore` object ready for I/O operations.
* **Returns:** `ts.Future[ts.TensorStore]`
* **Key Parameters:**
    * `spec`: A Python `dict` or `ts.Spec` object. This is the only required argument.
    * `open: bool`: (Default: `False`) If `True`, opens an existing store. An error is raised if it doesn't exist.
    * `create: bool`: (Default: `False`) If `True`, creates the store if it doesn't exist.
    * `delete_existing: bool`: (Default: `False`) If `True`, deletes the store and its contents before creating it.
    * `schema: ts.Schema`: When creating a new store (`create=True`), you can pass a schema object (often from another store) to define all its metadata at once (chunking, compression, dtype, etc.).
    * `dtype`, `domain`, `shape`, `chunk_layout`, `codec`, `fill_value`: Keywords to override specific schema properties when creating a store.
    * `transaction: ts.Transaction`: An existing transaction to associate with all operations on the opened store.
    * `context: ts.Context`: Specifies shared resources like cache pools or concurrency limits.

* **Example (Local Zarr):**
    ```python
    import tensorstore as ts
    import numpy as np

    # Define a spec for a local Zarr array
    spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': '/tmp/my_detailed_zarr'},
        'metadata': {
            'dtype': '|u1',  # uint8
            'shape': [100, 256, 256],
            'chunks': [16, 64, 64]
        }
    }

    # Open the store, creating it if it doesn't exist
    store_future = ts.open(spec, create=True, delete_existing=True)
    store = store_future.result()
    print(store)
    ```

### **`ts.cast(source, dtype)`**

Creates a virtual view of a `TensorStore` with a different data type.

* **Description:** `ts.cast` does not modify the underlying data. Instead, it creates a "view" that performs data type conversion on-the-fly as data is read or written. This is extremely memory and I/O efficient.
* **Returns:** A new `ts.TensorStore` view.
* **Example:**
    ```python
    # Assume 'store' is a uint8 TensorStore
    float_view = ts.cast(store, ts.dtype('float32'))
    print(f"Original dtype: {store.dtype}, View dtype: {float_view.dtype}")
    ```

### **`ts.downsample(source, downsample_factors, method)`**

Creates a view that downsamples the source data by aggregating blocks of data.

* **Description:** Useful for creating lower-resolution views of a large dataset for visualization or analysis without creating a new copy.
* **Parameters:**
    * `source`: The `ts.TensorStore` to downsample.
    * `downsample_factors`: A sequence of integers specifying the downsample factor for each dimension.
    * `method: str`: The aggregation method, e.g., `'mean'`, `'min'`, `'max'`, `'median'`, `'add'`.
* **Returns:** A new `ts.TensorStore` view.

### **`ts.lazy_driver(spec)`**
A driver that wraps another driver, deferring opening the base driver until the first operation. This is useful for improving performance when a large number of `TensorStore` objects may be opened but not all will be used.

---

## **2. Drivers**

A **driver** is a backend that handles reading and writing data in a specific format or to a specific storage system. The driver is specified in the `spec`.

### **Format Drivers**
These drivers define the on-disk format of a chunked array. They are specified in the top-level `driver` key of a `spec` and use a `kvstore` to determine the storage location.

* **`zarr` / `zarr3`**: A popular, modern format for chunked, compressed, N-dimensional arrays. It's language-agnostic and widely supported. `zarr3` is the newer version of the specification. This is the most common driver for new datasets.

* **`n5`**: Another common format for chunked N-dimensional arrays, widely used in microscopy and bio-imaging (e.g., by the Janelia Research Campus).

* **`neuroglancer_precomputed`**: The multi-resolution format used by Google's Neuroglancer for visualizing large 3D/4D datasets, common in neuroscience. It consists of multiple downsampled versions of the base data.

* **`array`**: A simple driver that stores an in-memory NumPy array. Useful for testing or creating small, temporary TensorStore objects.

### **Adapter Drivers**
These drivers wrap other drivers to add functionality.

* **`cast`**: Wraps another driver to change the data type on-the-fly.
* **`downsample`**: Wraps another driver to aggregate data and serve a lower-resolution view.
* **`stack`**: Stacks multiple source arrays along a new dimension.
* **`lazy`**: Wraps another driver to defer opening the base driver until the first I/O operation.

### **Specialized Drivers**

* **`ocdbt`**: The **Optimized Concurrent Database Transaction** driver. While technically a `kvstore` driver, it's often used as a high-performance *format* by libraries like **Orbax**.
    * **Use Case**: Designed for writing many small arrays concurrently and efficiently, which is a perfect match for saving ML model checkpoints (PyTrees). It aggregates many array chunks into a smaller number of database-like files, avoiding performance issues on cloud filesystems that are slow with many small files.
    * **Spec Structure**: When used as a format, it appears as the `driver` in the `kvstore` spec, which is then nested inside a `zarr` driver spec.
    * **Example (Orbax-style Checkpoint Array):**
        ```python
        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'ocdbt',
                'base': { 'driver': 'file', 'path': '/tmp/my_orbax_checkpoint/1' },
                'path': 'layer1.weights' # The specific array within the checkpoint
            }
        }
        ```
---

## **3. The `TensorStore` Object**

The `TensorStore` object is the central class you interact with. It represents an open, n-dimensional array and is the primary interface for all I/O and data manipulation operations. It's important to remember that this object is just a lightweight handle; the actual data resides on disk or in the cloud.

### **Reading and Writing**

I/O operations are performed on a `TensorStore` object (or a view of one) and are always asynchronous, returning a `Future`.

* **`.read(**kwargs) -> ts.Future[np.ndarray]`**: Reads the region represented by the `TensorStore` object and returns a `Future` that resolves to a NumPy array.
    * **`order: 'C' | 'F'`**: Specifies the memory layout of the returned NumPy array. `'C'` for C/row-major order (default), `'F'` for Fortran/column-major order.

* **`.write(source, **kwargs) -> ts.Future`**: Writes data to the region. This is the correct way to perform a copy.
    * **`source`**: The data to write. This can be a NumPy array, a Python scalar, or another `TensorStore` object.
    * **`order: 'C' | 'F'`**: If the source is another `TensorStore`, specifies the iteration order.

* **Example:**
    ```python
    # Assume 'store' is an open TensorStore
    
    # Create a 20x20 patch of data
    patch = np.ones((20, 20), dtype=np.uint8) * 255

    # Write the patch to a specific location
    write_future = store[4, 100:120, 100:120].write(patch)
    write_future.result()  # Wait for the write to complete

    # Read the same patch back
    read_future = store[4, 100:120, 100:120].read()
    read_patch = read_future.result()
    ```

### **Key Attributes**

* **`.dtype -> ts.dtype`**: The data type of the array elements (e.g., `ts.uint8`, `ts.float32`).
* **`.domain -> ts.IndexDomain`**: A rich object representing the full coordinate system, including the shape, dimension labels, and units.
* **`.rank -> int`**: The number of dimensions of the array. `ndim` is an alias for this.
* **`.shape -> tuple[int, ...]`**: A convenience property for `store.domain.shape`, returning the dimensions of the array as a tuple.
* **`.schema -> ts.Schema`**: A `Schema` object containing all structural metadata, including `dtype`, `domain`, `chunk_layout`, `codec`, `fill_value`, etc.
* **`.transaction -> ts.Transaction | None`**: The transaction this `TensorStore` handle is bound to, or `None` if it is not transactional.

### **Key Methods**

* **`.spec(minimal=False) -> ts.Spec`**: Returns a serializable `Spec` object that can be used to re-open this `TensorStore`. If `minimal=True`, it omits properties that can be inferred.
* **`.with_transaction(txn) -> ts.TensorStore`**: Returns a *new* view of the store that is bound to a specific transaction. All subsequent operations on this new view will be part of the transaction.
* **`.astype(dtype) -> ts.TensorStore`**: Alias for `ts.cast(store, dtype)`, creating a virtual view with a different data type.
* **`.resolve() -> ts.Future[ts.TensorStore]`**: For `lazy` drivers, this forces the underlying store to be opened. For regular stores, it re-validates cached metadata.

---

## **4. Indexing and Dimension Expressions**

Indexing in TensorStore is one of its most powerful features. Unlike NumPy, indexing a `TensorStore` does not return the data itself; instead, it creates a new, virtual `TensorStore` **view** into the data without copying or reading anything. All transformations are applied on-the-fly as data is read or written through the view. This approach is incredibly efficient for working with large arrays.

### **Basic Indexing (NumPy-style)**

You can use all the standard NumPy indexing semantics you are familiar with.

* **Integer:** `store[5]`
    * Selects a single slice, reducing the rank of the view by one.

* **Slice:** `store[10:20:2]`
    * Selects a range with a given start, stop, and step.

* **Ellipsis (`...`):** `store[0, ..., 0]`
    * A placeholder for as many full slice dimensions (`:`) as needed. Useful for selecting the outer boundaries of a high-dimensional array.

* **New Axis (`np.newaxis` or `None`):** `store[..., np.newaxis]`
    * Adds a new dimension of size 1, increasing the rank of the view.

### **Advanced Indexing**

* **Index Arrays:** Select elements at specific coordinates using arrays of integers. The shape of the resulting view depends on the shape of the index arrays.
    ```python
    # Selects elements at coordinates (0, 10) and (5, 20)
    # The result is a 1-D view of rank 1 and shape (2,)
    points = store[[0, 5], [10, 20]]
    ```

* **Boolean Indexing:** Select elements where a boolean array is `True`. The boolean array must be broadcastable to the shape of the indexed dimensions.
    ```python
    # Create a boolean mask for the first dimension
    mask = np.array([True, False, True, ...]) # Length must match the dimension size
    selected_slices = store[mask]
    ```

### **Dimension Expressions (`ts.d`)**

This is the most expressive and powerful method for indexing in TensorStore. It allows you to refer to dimensions by index (`ts.d[i]`) or by their string label and apply complex transformations.

* `ts.d[i]`: Refers to dimension `i`.
* `ts.d['name']`: Refers to a dimension by its label (if defined in the `IndexDomain` schema).

#### **Key Transformations on Dimensions:**

* **`.label('new_name')`**: Adds or changes a dimension's label in the output view.
* **`.translate_by(offset)`**: Shifts the origin (starting coordinate) of a dimension's domain. `store[ts.d[0].translate_by(50)]` makes index `0` of the view correspond to index `50` of the original store.
* **`.stride(factor)`**: Applies a stride, equivalent to `::factor`.
* **`.broadcast`**: Adds a new dimension of size 1 (similar to `np.newaxis`).
* **`.transpose(*labels_or_indices)`**: Reorders dimensions to match the given new order.

#### **Example of a Dimension Expression:**

Imagine a `(C, H, W)` image where `C` is the channel. You want to extract the first channel (`C=0`), flip the image vertically, and relabel the dimensions for clarity.

```python
# Assume an initial store with shape [3, 100, 200]
# and dimension labels ['c', 'y', 'x']

# Create a transformed view using dimension expressions
view = store[
    ts.d['c'] == 0,              # Select the first channel, removing the dimension
    ts.d['y'][::-1],             # Reverse the 'y' dimension (flips vertically)
    ts.d['x'].label('width')     # Relabel the 'x' dimension to 'width'
]

# The resulting view has shape (100, 200) and labels ('y', 'width')
# and its y-axis is inverted relative to the source.
# All of this is done without copying any data.
```
---

## **5. `ts.Spec` - The Specification Object**

A `ts.Spec` is a serializable object, based on a Python `dict`, that contains all the information needed to open or create a `TensorStore`. It's the primary mechanism for configuring data access and is essential for reproducibility.

* **Description:** The `Spec` object encapsulates the driver, storage location (`kvstore`), and all metadata. Because it can be easily converted to and from JSON, it's ideal for saving configurations, passing them between processes, or storing them in a database. You can interact with it like a normal dictionary.

* **Creation:**
    * `ts.Spec(spec_dict)`: Creates a `Spec` object from a dictionary.
    * `store.spec()`: Retrieves the `Spec` from an existing `TensorStore` object.

* **Key Methods:**
    * `.to_json()`: Serializes the spec to a JSON-compatible `dict`. This is useful for storage or transmission.
    * `.update(other_spec)`: Merges another spec into the current one, similar to `dict.update`.

* **Example:**
    ```python
    # Assume 'spec_dict' is the dictionary from the first section
    spec_dict = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': '/tmp/my_detailed_zarr'}
    }

    # Create a spec object from a dictionary
    spec_obj = ts.Spec(spec_dict)

    # Modify the spec like a dictionary, e.g., to open in read-only mode
    spec_obj['open'] = True
    spec_obj['create'] = False
    spec_obj['recheck_cached_data'] = 'open' # Re-check data on open

    # You can also update it from another dictionary
    spec_obj.update({'context': {'cache_pool': {'total_bytes_limit': 100000000}}})


    # Open the store using the modified spec object
    # Note: this will fail if the store doesn't already exist
    # store_ro = ts.open(spec_obj).result()

    # Print the JSON representation
    print(spec_obj.to_json())
    ```
---

## **6. `ts.Schema` - The Data Structure Definition**

A `Schema` describes the structural properties of a `TensorStore`—the "what"—without defining its storage location—the "where". This makes it perfect for creating new arrays with the same structure as existing ones.

* **Description:** The `Schema` object is a container for all the metadata that defines an array's shape, data type, and on-disk representation. It can be retrieved from an existing store and used to create a new store with an identical layout but at a different location.

* **Key Attributes:**
    * `.dtype -> ts.dtype`: The data type of the elements.
    * `.domain -> ts.IndexDomain`: The coordinate system, including shape and dimension labels.
    * `.rank -> int`: The number of dimensions.
    * `.chunk_layout -> ts.ChunkLayout`: The chunking scheme, including the shape of each chunk.
    * `.codec -> ts.CodecSpec`: The compression and filter pipeline (e.g., `blosc`, `zstd`, `gzip`).
    * `.fill_value`: The value returned for parts of the array that have not been written.
    * `.dimension_units -> tuple[ts.Unit | None]`: The physical units associated with each dimension (e.g., `'nm'`).

* **Example (Copying a schema to create a structurally identical store):**
    ```python
    # Assume `source_store` is an existing TensorStore
    
    # Define a new location for our destination store
    dest_spec = {
        'driver': 'zarr',
        'kvstore': {'driver': 'file', 'path': '/tmp/my_zarr_copy'}
    }

    # Create the destination store using the source's schema
    # This preserves chunking, compression, dtype, shape, etc.
    dest_store = ts.open(
        dest_spec,
        create=True,
        delete_existing=True,
        schema=source_store.schema
    ).result()

    print(f"New store created with chunk layout: {dest_store.chunk_layout}")
    print(f"New store created with codec: {dest_store.codec}")
    ```

---

## **7. `ts.KvStore` - Key-Value Storage API**

`KvStore` provides a low-level interface that treats any storage system as a simple key-value dictionary, where keys are file paths (strings) and values are their raw content (bytes). This is the fundamental building block that all format drivers (like `zarr`) use to interact with storage.

* **Description:** This API is useful for low-level inspection, debugging, or directly manipulating the individual files within a storage location without interpreting them as a structured array.

* **`ts.KvStore.open(spec) -> ts.Future[ts.KvStore]`**: Opens a Key-Value Store. The `spec` must define a `driver` and location information.
    * **Common `KvStore` Drivers:** `file` (local filesystem), `gcs` (Google Cloud Storage), `s3` (Amazon S3), `memory`, and `ocdbt`.

* **`KvStore` Object Methods:**
    * `.read(key, **kwargs) -> ts.Future`: Reads the bytes for a given key.
    * `.write(key, value) -> ts.Future`: Writes bytes to a key.
    * `.delete(key) -> ts.Future`: Deletes a key-value pair.
    * `.list(**kwargs) -> ts.Future`: Lists all keys in the store. Returns a `Future` that resolves to a list of byte strings.

* **Example:**
    ```python
    # Open the local filesystem as a KvStore
    fs_kvstore_future = ts.KvStore.open({'driver': 'file', 'path': '/tmp/kv_test/'})
    fs_kvstore = fs_kvstore_future.result()

    # Write a file
    fs_kvstore.write('hello.txt', b'Hello, World!').result()

    # Read it back
    content_future = fs_kvstore.read('hello.txt')
    # The result of a read has a `.data` attribute
    print(content_future.result().data)

    # List contents
    keys = [key.decode() for key in fs_kvstore.list().result()]
    print(f"Keys found: {keys}")
    ```
---

## **8. `ts.Transaction`**

`ts.Transaction` provides a mechanism to ensure a set of read and write operations are **atomic**, meaning they are applied as a single, indivisible unit. This is critical for preventing data corruption when multiple processes might be trying to modify the same data concurrently (a "race condition").

* **Description:** Transactions in TensorStore use an **optimistic concurrency** model. When you perform reads and writes within a transaction, the operations are staged locally. When you commit, TensorStore first checks if the underlying data has been modified by another process since you first read it. If it has, the commit fails with an error, and you can choose to retry the entire transaction. This is highly efficient as it avoids locking the data for long periods.

* **Lifecycle:**
    1.  `txn = ts.Transaction()`: Create a new, independent transaction object.
    2.  `store_in_txn = store.with_transaction(txn)`: Bind a `TensorStore` handle to the transaction. All subsequent operations on `store_in_txn` are part of this transaction.
    3.  Perform your sequence of read and write operations on `store_in_txn`.
    4.  `txn.commit_async().result()`: Atomically commit all staged changes. This will fail if a conflict is detected.
    5.  `txn.abort()`: If an error occurs or you wish to cancel the changes, you can abort the transaction.

* **Example (Atomic Increment):**
    ```python
    # Assume 'store' is open and contains integer data
    pixel_coord = (10, 20, 30)
    txn = ts.Transaction()

    try:
        # Bind the store to the transaction
        store_in_txn = store.with_transaction(txn)

        # Perform the read-modify-write sequence
        current_value = store_in_txn[pixel_coord].read().result()
        store_in_txn[pixel_coord].write(current_value + 1).result()

        # Atomically commit the changes
        txn.commit_async().result()
        print("Transaction committed successfully.")

    except Exception as e:
        print(f"Transaction failed, aborting: {e}")
        # It's good practice to explicitly abort on failure
        txn.abort()
    ```

---

## **9. `ts.Future`**

A `Future` is an object that acts as a placeholder for the result of an asynchronous operation. Because all I/O in TensorStore is non-blocking, every function that performs I/O (like `.read()`, `.write()`, or `ts.open()`) returns a `Future` immediately.

* **Description:** This allows your program to queue up many I/O operations without waiting for each one to complete. You can continue with other computations and then retrieve the result from the `Future` when you actually need it.

* **Key Methods:**
    * `.result(timeout=None)`: Blocks execution until the operation is complete and returns the result. If the operation failed, this method will raise the corresponding exception. This is the most common way to interact with a `Future`.
    * `.done() -> bool`: Returns `True` if the operation has completed (either successfully or with an error). This is useful for non-blocking checks.
    * `.exception() -> Exception | None`: Returns the exception object if the operation failed, otherwise returns `None`. This allows you to check for errors without raising them.
    * `.add_done_callback(fn)`: Attaches a callback function `fn` that will be executed when the future completes. The function will be called with the `Future` object as its only argument.

* **Example:**
    ```python
    # Assume 'store' is an open TensorStore
    
    # This operation is dispatched to run in the background
    future = store[0, 0:10, 0:10].read()
    
    # You can do other work here while the read is happening...
    print("Read operation has been queued...")
    
    # Now, block and get the result when you need it
    data = future.result()
    print("Read operation complete. Data shape:", data.shape)
    ```
---

## **10. `ts.Context` - Managing Shared Resources**

A `ts.Context` is an object that holds shared, process-wide resources that can be used by multiple `TensorStore` and `KvStore` instances. It's the primary way to control caching, concurrency, and other performance-tuning parameters.

* **Description:** By default, TensorStore uses a global default context. However, for fine-grained control, you can create your own `Context` object and pass it to `ts.open()`. This is useful in applications with complex I/O patterns or when you need to isolate the caching behavior of different parts of your program.

* **Creation:**
    ```python
    context = ts.Context({
        'cache_pool': {'total_bytes_limit': 100000000}, # 100MB cache
        'data_copy_concurrency': {'limit': 16},
        'file_io_concurrency': {'limit': 64}
    })
    
    # Use the context when opening a store
    store_with_context = ts.open(spec, context=context).result()
    ```

* **Key Resources You Can Configure:**
    * **`cache_pool`**: Controls the size of the in-memory cache for data chunks.
    * **`data_copy_concurrency`**: Limits the number of concurrent data copy operations.
    * **`file_io_concurrency`**: Limits the number of concurrent file I/O threads.

---

## **11. `ts.IndexDomain` - The Coordinate System**

An `IndexDomain` is a rich object that represents the coordinate system of a `TensorStore`. It's more powerful than a simple `shape` tuple because it includes dimension labels, bounds, and units.

* **Description:** Every `TensorStore` has a `.domain` attribute. Understanding this object is key to mastering TensorStore's indexing capabilities, as all indexing operations are essentially transformations applied to the `IndexDomain`.

* **Key Attributes:**
    * `.shape -> tuple[int, ...]`**: The size of each dimension.
    * `.rank -> int`**: The number of dimensions.
    * `.labels -> tuple[str, ...]`**: The string label for each dimension.
    * `.inclusive_min -> tuple[int, ...]`**: The starting coordinate for each dimension.
    * `.exclusive_max -> tuple[int, ...]`**: The ending coordinate for each dimension.

* **Example:**
    ```python
    # Assume 'store' is an open TensorStore
    domain = store.domain
    
    print(f"Shape: {domain.shape}")
    print(f"Rank: {domain.rank}")
    print(f"Labels: {domain.labels}")
    print(f"Bounds: {domain.bounds}")
    
    # You can also create a domain from scratch
    my_domain = ts.IndexDomain(
        labels=['channel', 'y', 'x'],
        shape=[3, 1024, 1024],
        inclusive_min=[0, -512, -512] # Center the origin
    )
    print("\nCustom Domain:", my_domain)
    ```

---
---

## **Appendix: Official Documentation Links**

For even more detail, the official documentation is the ultimate source of truth.

* **[Python API Index](https://google.github.io/tensorstore/python/api/index.html)**: The top-level landing page for the Python API.
* **[Indexing and Dimension Expressions](https://google.github.io/tensorstore/python/indexing.html)**: A detailed guide on all indexing features.
* **[Driver Reference](https://google.github.io/tensorstore/driver/index.html)**: A complete list of all format and adapter drivers.
* **[KvStore Reference](https://google.github.io/tensorstore/kvstore/index.html)**: A complete list of all Key-Value Store drivers, including `ocdbt`.