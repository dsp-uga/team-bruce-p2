# CSCI 8360 Data Science Practicum : Cilia Segmentation

  
  ## Getting Started
  These instructions describe the prerequisites and steps to get the project up and running.
  
  ### Prerequisites
  This project can be easily set up on the Google Cloud Platform, using a '[Deep Learning VM Instance](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning)'. You will need to have [Google Cloud SDK](https://cloud.google.com/sdk/install) installed in your local machine to be able to set the project up. 
  
  After downloading/cloning this repository to your local machine, the user will need to open the `Google Cloud SDK Shell`. Once it opens, the user can copy the contents of this repository to the Deep Learning VM instance using the command:
  
  `gcloud compute scp --recurse /complete/link/to/repository/* <user>@<instance_name-vm>:/home/<user>/`
  
  The Deep Learning VM instance is a good place to deploy this project, because it comes pre-installed with a majority of the packages used in this project, such as, 'OpenCV', 'Tensorflow', 'Keras', 'Matplotlib', 'Sklearn', 'Pandas', 'PIL', 'Numpy'. The packages that are installed through the DataLoader class upon spinning up the project include 'Tensorly'. 
    
  ### Usage
  To run the code and generate output prediction masks in the `/results` directory, the user can navigate to the folder containing the file 'team-bruce.py', and run it using the command: `python team-bruce.py --options`. The user can get a description of the options by using the command `python team-bruce.py -help`.

  ### Output
  Upon running the command in the ‘Usage’ section, the dataset will be downloaded from the Google Storage bucket link carrying the Cilia dataset, and the output prediction masks will be generated in the `/results` directory.
  
  Considering our inexperience in dealing with images, and the challenges/ambiguity in the Cilia dataset, our Autolab submissions were, in general, pretty average and well under our expectations, for the variety of models that we implemented through the course of this project.

## Contributors
* See [Contributors](https://github.com/dsp-uga/team-bruce-p2/blob/master/CONTRIBUTORS.md) file for more details.

## License
This project is licensed under the **MIT License**. See [LICENSE](https://github.com/dsp-uga/team-bruce-p2/blob/master/LICENSE) for more details.
