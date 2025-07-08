# TensorStore & Orbax: From Basics to Advanced Checkpointing 🚀

This repository contains a hands-on Google Colab notebook designed to teach you **TensorStore** from the ground up, culminating in a practical understanding of how it powers **Orbax** for efficient machine learning checkpointing.

This tutorial was built interactively and contains a complete, battle-tested walkthrough that addresses common errors and nuances of the TensorStore API.

---
## 📖 What You'll Learn

This notebook is a step-by-step guide that explains not just the "how," but also the "why." By the end, you will understand:

* **The Problem with Big Data:** Why tools like NumPy fall short and why TensorStore is necessary.
* **Core TensorStore Mechanics:** How to read and write large, out-of-memory arrays using `specs`, `drivers`, and `kvstores`.
* **Asynchronous Operations:** How TensorStore uses `Future` objects for high-performance I/O.
* **Cloud Integration:** How to use the same code to work with data on your local disk or in the cloud (e.g., GCS).
* **Advanced Features:** How to use virtual views for zero-cost transformations and transactions for data integrity.
* **Orbax Integration:** How Orbax uses TensorStore's specialized `ocdbt` driver to efficiently save and restore JAX model checkpoints.
* **Practical Skills:** How to manually inspect and convert Orbax checkpoints using the TensorStore library.

---
## 🚀 How to Use

1.  Click on the "Open in Colab" badge at the top of this README.
2.  The notebook will open in Google Colab.
3.  Run the cells sequentially from top to bottom to follow the tutorial.

---
## 🧠 Core Concepts Covered

* **TensorStore:**
    * `specs`, `drivers`, `kvstores`
    * Chunking and Asynchronous I/O
    * Zarr and OCDBT drivers
    * Virtual Views and Transactions
    * The `KvStore` API for low-level inspection
* **Orbax:**
    * `StandardCheckpointer` for saving/restoring PyTrees.
    * Understanding the on-disk format of an Orbax checkpoint.
    * Converting checkpoints to standard formats for interoperability.